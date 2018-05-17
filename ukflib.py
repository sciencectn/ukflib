from __future__ import print_function
import numpy as np
import scipy.linalg
import math
from math import pi
import sys
from numpy.linalg import norm
import random

class FilterError(Exception):
    pass

def _check_covariance(M,output=True):
    """
    Check if the matrix is a valid covariance matrix
    It should be positive definite
    :param M:
    :return:
    """
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        if output:
            print(e, file=sys.stderr)
        return False
    return True

def angular_fix(a):
    fix = np.mod(np.mod(a,2*pi)+2*pi,2*pi)
    if np.isscalar(fix):
        if fix > pi:
            fix -= 2*pi
    else:
        fix[fix > pi] -= 2*pi
    return fix

def repair_covariance(P):
    assert not np.isnan(P).any()
    R = P.copy()
    i=0
    while not _check_covariance(R,False):
        if i > 200:
            # For some weird reason, this function fails to
            # repair certain matrices when running in parallel.
            # My guess is that eigh or cholesky are not reentrant.
            print(np.linalg.eigh(R)[0],file=sys.stderr)
            e, v = np.linalg.eigh(R)
            R = _repair_bad_eig(R,e,v)
            print(np.linalg.eigh(R)[0],file=sys.stderr)
            print(R,file=sys.stderr)
            raise FilterError("Cannot repair matrix")
        R = (R + R.T)/2.0    # Force symmetry
        e,v = np.linalg.eigh(R)
        R = _repair_bad_eig(R,e,v)  # Remove a bad eigenvalue
        i += 1
    # print(i)
    return R

def _repair_bad_eig(P,eigs,vecs,new_eig=0.01):
    """
    Remove negative eigenvalues from a matrix
    Change the new eigenvalue to 1e-3
    :param P:
    :return:
    """
    for i,e in enumerate(eigs):
        if e <= 1e-7:
            v = vecs[:,i]
            v /= norm(v)
            P = P + np.outer(v,v)*(abs(e) + random.uniform(1e-3,new_eig))
    return P


class UnscentedKalmanFilter(object):
    def __init__(self,
                 state_size,
                 process_noise,
                 measurement_noise,
                 init_covariance=None,
                 init_state=None,
                 angle_mask=None,
                 kappa=0,
                 alpha=1.0,
                 beta=2,
                 repair_covariance=False):
        assert 0 < alpha <= 1,\
            "alpha needs to be in range (0,1], got {0}".format(alpha)
        self._repair_covar = repair_covariance
        self._ss = state_size
        self._state = np.zeros(state_size)
        if init_state is not None:
            init_state = np.atleast_1d(init_state)
            assert len(init_state)==state_size,\
                "State size does not match initial state"
            self._state[:] = init_state

        if init_covariance is None:
            init_covariance = np.eye(self._ss)
        else:
            init_covariance = np.atleast_2d(init_covariance)
            assert init_covariance.shape[0]==state_size,\
                "The initial covariance should have the same dimension as the state size"

        self._angle_mask = None
        if angle_mask is not None:
            assert len(angle_mask) == self._ss, \
                "The angle mask should be the same length as the state size"
            self._angle_mask = (np.atleast_1d(angle_mask) != 0)

        process_noise = np.atleast_2d(process_noise)
        measurement_noise = np.atleast_2d(measurement_noise)

        if not _check_covariance(process_noise):
            raise FilterError("Invalid process noise covariance")
        if not _check_covariance(measurement_noise):
            raise FilterError("Invalid measurement noise covariance")
        if not _check_covariance(init_covariance):
            raise FilterError("Invalid initial state covariance")

        # Process and measurement noise state sizes
        self._sv = process_noise.shape[0]
        self._sn = measurement_noise.shape[0]

        # Augmented state size
        self._sa = self._ss + self._sv + self._sn

        # Augmented covariance matrix
        self._Pa = scipy.linalg.block_diag(init_covariance, process_noise, measurement_noise)

        # This is the "lambda" value from Van der Merwe's thesis
        self._lambda = alpha ** 2 * (self._sa + kappa) - self._sa

        # Weights for mean and covariance computation
        self._weight_m = np.zeros(2*self._sa+1)
        self._weight_m[:] = 1 / (2 * (self._sa + self._lambda))
        self._weight_m[0] = self._lambda / (self._lambda + self._sa)
        self._weight_c = self._weight_m.copy()
        self._weight_c[0] += (1 - alpha**2 + beta)

        self._sigma_pts = np.zeros((self._sa, 2*self._sa + 1))
        self._get_sigma_points()
        self._P_state = "none"


    def predict(self, predict_fn, *args):
        """
        The prediction step of the UKF, also referred to as
        the "time-update" or "propagation" step

        Given a callback function which returns the next step,
        update the mean and covariance of the filter

        :param predict_fn: A callback which takes the current state,
                    a sampling of the noise, and any user arguments
                    and returns the next state.
                    It has the form:
                    predict_fn(state, noise, *args)
                    where 'state' is an ndarray of the current state,
                    'noise' contains a sampling of the process noise random variables,
                    and *args contains any optional user args
        :param args: Optional user arguments to be passed to the callback
        :return:
        """
        self._get_sigma_points()
        for j in range(self._sigma_pts.shape[1]):
            sigma_pt = self._sigma_pts[:,j]
            state = sigma_pt[:self._ss]
            noise = sigma_pt[self._ss:self._ss+self._sv]
            new_state = predict_fn(state, noise, *args)
            self._sigma_pts[:self._ss,j] = new_state
        self._update_mean_and_covariance()
        self._P_state = "minus"

    def update(self, update_fn, measurement, *args):
        """
        The measurement update

        Given a measurement callback function and the current measurement,
        update the mean and covariance

        :param update_fn: A callback which takes the current state, a sampling
                    of the measurement noise, and any optional arguments and
                    returns the predicted measurement. It has the form:
                    update_fn(state, noise, *args)
                    where state is the current state, noise is a sampling of the
                    measurement noise random variables, and *args contains any
                    optional user arguments.
        :param measurement:
        :param args:
        :return:
        """
        if self._P_state=="plus":
            # Any nonlinearities from the predict step will be lost if you update twice in a row
            # Maybe someone out there really wants to do this but I'm going to assume it's a bug.
            raise FilterError("You cannot call update twice in a row. "
                              "Multiple updates should be combined into one measurement model. ")


        Z = None
        for j in range(self._sigma_pts.shape[1]):
            sigma_pt = self._sigma_pts[:,j]
            state = sigma_pt[:self._ss]
            noise = sigma_pt[self._ss+self._sv:]
            zj = update_fn(state, noise, *args)
            if Z is None:
                Z = np.zeros((len(zj), self._sigma_pts.shape[1]))
            Z[:,j] = zj
        zbar = Z.dot(self._weight_m)
        Pz = self._sigma_pt_covariance(Z,zbar,fix_angles=False)
        _check_covariance(Pz)
        Pxz = self._sigma_pt_covariance(self._sigma_pts[:self._ss,:],self._state,
                                        pts2=Z,mean2=zbar)
        K = Pxz.dot(np.linalg.inv(Pz))
        innovation = measurement - zbar
        self._state += K.dot(innovation)
        self._Pa[:self._ss,:self._ss] -= K.dot(Pz).dot(K.T)
        self._P_state = "plus"
        return innovation


    def _get_sigma_points(self):
        """
        Use the Cholesky decomposition to get the sigma points
        :return:
        """
        try:
            M = np.linalg.cholesky(self._Pa)
        except np.linalg.LinAlgError:
            if not self._repair_covar:
                raise FilterError(
                    "The covariance is not positive definite (most likely the filter is overconfident)."
                    " Use the option repair_covariance=True to force positive definiteness or increase your noise.")
            self._Pa[:self._ss,:self._ss] = repair_covariance(self._Pa[:self._ss,:self._ss])
            assert not np.isnan(self._Pa).any()
            try:
                M = np.linalg.cholesky(self._Pa)
            except np.linalg.LinAlgError as e:
                print(e, file=sys.stderr)
                print(self._Pa, file=sys.stderr)


        self._sigma_pts.fill(0)
        for j in range(2*self._sa+1):
            self._sigma_pts[:self._ss,j] = self._state
        gamma = math.sqrt(self._sa + self._lambda)
        self._sigma_pts[:,1:self._sa+1] += gamma*M
        self._sigma_pts[:,self._sa+1:] -= gamma*M
        self._need_sigma_pts = False


    def _update_mean_and_covariance(self):
        """
        Assuming we have modified the sigma points, recompute
        the mean and covariance of the state estimate
        :return:
        """

        ## Mean
        # Normalize all the angles in the sigma points
        if self._angle_mask is not None:
            states = self._sigma_pts[:self._ss,:]
            states[self._angle_mask, :] = angular_fix(states[self._angle_mask, :])
            for i in range(self._ss):
                row = self._sigma_pts[i, :]
                if self._angle_mask[i]:
                    # Vectorize the angles first, then find the weighted mean
                    c = np.cos(row).dot(self._weight_m)
                    s = np.sin(row).dot(self._weight_m)
                    self._state[i] = math.atan2(s,c)
                else:
                    # Just a normal weighted mean
                    self._state[i] = row.dot(self._weight_m)
        else:
            self._state = self._sigma_pts[:self._ss,:].dot(self._weight_m)

        ## Covariance
        self._Pa[:self._ss,:self._ss] = \
            self._sigma_pt_covariance(self._sigma_pts[:self._ss,:], self._state)


    def _sigma_pt_covariance(self, pts, mean, fix_angles=True, pts2=None, mean2=None):
        cov = None
        for j in range(pts.shape[1]):
            diff = pts[:,j] - mean
            if fix_angles:
                self._ang_fix(diff)
            if pts2 is not None:
                diff2 = pts2[:,j] - mean2
            else:
                diff2 = diff
            o = self._weight_c[j]*np.outer(diff,diff2)
            if cov is None:
                cov = o
            else:
                cov += o
        return cov

    def _ang_fix(self, v):
        """
        Fix the angles in a vector to -pi..pi
        This operates in place
        :return:
        """
        if self._angle_mask is None:
            return v
        v[self._angle_mask] = angular_fix(v[self._angle_mask])
        return v

    @property
    def state_size(self):
        return self._ss

    @property
    def covariance(self):
        """
        The covariance of the estimation error, i.e.
        cov[ (X-X_hat) ]
        X is the state and X_hat is the estimated state
        and both are random variables.
        :return:
        """
        return self._Pa[:self._ss,:self._ss].copy()

    @property
    def process_noise(self):
        return self._Pa[self._ss:self._ss+self._sv, self._ss:self._ss+self._sv].copy()

    @property
    def measurement_noise(self):
        return self._Pa[self._ss+self._sv:, self._ss+self._sv:].copy()

    @property
    def state(self):
        return self._state.copy()

    @property
    def angle_mask(self):
        if self._angle_mask is None:
            return None
        return self._angle_mask.copy()

    @state.setter
    def state(self, x):
        self._state[:] = x

