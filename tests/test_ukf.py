

import nose.tools
from nose.tools import with_setup
import ukflib
import numpy as np
import numpy.testing as npt
import random
from numpy.linalg import inv

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


