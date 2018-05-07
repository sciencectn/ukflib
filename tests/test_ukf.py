

import nose.tools
from nose.tools import with_setup, assert_true
import ukflib
import numpy as np
import numpy.testing as npt
import random
from numpy.linalg import inv
from math import pi

def random_covariance(n):
    pts = np.random.rand(n,n+1)
    return np.cov(pts)

def setup():
    random.seed(0)
    np.random.seed(0)

@with_setup(setup)
def test_constructor():
    """
    Verify that internal variables are getting set correctly
    :return:
    """
    Rv = random_covariance(3)    # Process noise
    Rn = random_covariance(3)    # Measurement noise
    P0 = random_covariance(3)    # Init covariance
    X0 = np.random.randn(3)      # Init state
    ukf = ukflib.UnscentedKalmanFilter(3,Rv,Rn,P0,X0)
    npt.assert_almost_equal(ukf.covariance,P0)
    npt.assert_almost_equal(ukf.state,X0)
    npt.assert_almost_equal(ukf.process_noise, Rv)
    npt.assert_almost_equal(ukf.measurement_noise,Rn)

@with_setup(setup)
def test_simple_propagation_1d():
    """
    Test some simple linear 1D propagation functions
    :return:
    """
    # Create scalars to test constructor
    Rv = 3.0
    Rn = 2.0
    P0 = 1.0
    X0 = 0.0
    ukf = ukflib.UnscentedKalmanFilter(1,Rv,Rn,P0,X0)
    npt.assert_almost_equal(ukf.state, X0)
    npt.assert_almost_equal(ukf.covariance, P0)
    ukf.predict(lambda x,v: x+v)
    npt.assert_almost_equal(ukf.covariance, P0+Rv)
    npt.assert_almost_equal(ukf.state, X0)

    npt.assert_almost_equal(ukf.process_noise, Rv)
    npt.assert_almost_equal(ukf.measurement_noise, Rn)
    ukf.predict(lambda x,v: 2*x+v)
    npt.assert_almost_equal(ukf.state, X0)
    npt.assert_almost_equal(ukf.covariance, 4*(P0+Rv) + Rv)

@with_setup(setup)
def test_propagation_nd():
    """
    Test a more complicated (but linear) propagation function
    :return:
    """
    Rv = random_covariance(3)
    Rn = random_covariance(1)
    P0 = random_covariance(3)
    ukf = ukflib.UnscentedKalmanFilter(3,Rv,Rn,P0)
    A = np.random.rand(3,3)
    ukf.predict(lambda x,v: A.dot(x) + v)
    expected = A.dot(P0).dot(A.T) + Rv
    npt.assert_almost_equal(ukf.covariance, expected)

@with_setup(setup)
def test_kalman_filter_equivalence():
    """
    Ensure that the update equations match those of the linear Kalman Filter
    :return:
    """
    Rv = random_covariance(3)
    Rn = random_covariance(2)
    P0 = random_covariance(3)
    X0 = np.random.randn(3)
    ukf = ukflib.UnscentedKalmanFilter(3,Rv,Rn,P0,X0)

    for i in range(10):
        P = ukf.covariance
        x = ukf.state
        A = np.random.rand(3,3)
        ukf.predict(lambda x,v: A.dot(x) + v)
        expected = A.dot(P).dot(A.T) + Rv
        npt.assert_almost_equal(ukf.covariance, expected)
        npt.assert_almost_equal(ukf.state, A.dot(x))

        # Save these for comparison with the Kalman filter
        P_m = ukf.covariance
        x_m = ukf.state

        # Make a fake measurement and run an update
        H = np.random.rand(2,3)
        z = np.random.randn(2)
        ukf.update(lambda x,n: H.dot(x) + n, z)

        # Do a normal Kalman filter update

        K = P_m.dot(H.T).dot(inv(H.dot(P_m).dot(H.T) + Rn))
        x_p = x_m + K.dot(z - H.dot(x_m))
        P_p = (np.eye(3) - K.dot(H)).dot(P_m)

        npt.assert_almost_equal(ukf.state, x_p)
        npt.assert_almost_equal(ukf.covariance, P_p)

@with_setup(setup)
def test_angle_fixing():
    """
    Randomly add multiples of 2*pi to an angle and verify that nothing happens
    :return:
    """
    Rv = 1.337e-2
    Rn = random_covariance(1)
    P0 = 1.2345e-3
    ukf = ukflib.UnscentedKalmanFilter(1,Rv,Rn,
                                       init_covariance=P0,
                                       init_state=0,
                                       angle_mask=[1])
    ukf.predict(lambda x,v: x+0.01+random.randint(-10,10)*2*pi + v)
    npt.assert_almost_equal(ukf.state, 0.01)
    npt.assert_almost_equal(ukf.covariance, P0 + Rv)


def test_repair():
    P = np.array([[ 9.12986704e+01, -7.57679187e+01,  6.11720894e+00,
        -2.83243302e-13,  2.83243302e-13],
       [-7.57679187e+01,  7.54673700e+01, -5.52386361e+00,
         2.63664279e-13, -2.63664279e-13],
       [ 6.11720894e+00, -5.52386361e+00,  4.26759229e-01,
         5.16773336e-13, -5.16773336e-13],
       [-2.83243302e-13,  2.63664279e-13,  5.16773336e-13,
         5.00000000e-04,  5.00000000e-04],
       [ 2.83243302e-13, -2.63664279e-13, -5.16773336e-13,
         5.00000000e-04,  5.00000000e-04]])
    e,_ = np.linalg.eig(P)
    R = ukflib.repair_covariance(P)
    assert_true(ukflib._check_covariance(R))

def test_repair2():
    P = np.array([[ 2.21964059e+01, -6.46086830e+00, -3.73682378e-01,
         1.26620088e-15,  1.26620088e-15],
       [-6.46086830e+00,  7.94576601e+00,  9.70006895e-01,
         5.33428567e-16,  5.33428567e-16],
       [-3.73682378e-01,  9.70006895e-01,  1.64113280e-01,
         1.06252660e-16,  1.06252660e-16],
       [ 1.26620088e-15,  5.33428567e-16,  1.06252660e-16,
         5.00000000e-04, -5.00000000e-04],
       [ 1.26620088e-15,  5.33428567e-16,  1.06252660e-16,
        -5.00000000e-04,  5.00000000e-04]])
    R = ukflib.repair_covariance(P)
    assert_true(ukflib._check_covariance(R))

def test_repair3():
    P = np.array([[ 2.61213400e+01,  7.80928531e+00, -1.59125342e+00,
        -4.72454921e-16,  3.62525890e-01],
       [ 7.80928531e+00,  1.44902186e+01, -1.98094103e+00,
         6.70620912e-16, -4.69545053e-01],
       [-1.59125342e+00, -1.98094103e+00,  1.13024384e+00,
        -2.29290730e-16,  4.49370398e-01],
       [-4.72454921e-16,  6.70620912e-16, -2.29290730e-16,
         1.07606743e-30, -1.94888240e-15],
       [ 3.62525890e-01, -4.69545053e-01,  4.49370398e-01,
        -1.94888240e-15,  3.37873740e+00]])
    R = ukflib.repair_covariance(P)
    assert_true(ukflib._check_covariance(R))

    P = np.array([[  8.57580365,  -9.87047457 ,  8.41110293 , -0.51355479 ,  0.34236986],
                 [ -9.87047457,  17.25046354 ,-13.80590903 ,  0.2247901  , -0.14986007],
                 [  8.41110293, -13.80590903 , 11.71672657 , -1.72598453 ,  1.15065635],
                 [ -0.51355479,   0.2247901  , -1.72598453 ,  4.94716751 , -3.29811168],
                 [  0.34236986,  -0.14986007 ,  1.15065635 , -3.29811168 ,  2.19874112]])
    R = ukflib.repair_covariance(P)
    assert_true(ukflib._check_covariance(R))

