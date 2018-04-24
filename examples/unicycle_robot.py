

import ukflib
import numpy as np
from math import sin,cos, atan2, sqrt
from numpy.linalg import norm
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import random

def rotate(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]])

def get_displacement(ts, v, omega, dt):
    w = np.zeros(3)
    if omega < 1e-5:
        w[0] = v*dt
        w[0:2] = rotate(ts).dot(w[0:2])
    else:
        r = v/omega
        w[0] = -r*sin(ts) + r*sin(ts+omega*dt)
        w[1] = r*cos(ts) - r*cos(ts+omega*dt)
        w[2] = omega*dt
    return w


def predict_icc(state, noise, v, w, dt):
    """
    Approximate the next state of the robot's motion using
    the "instantaneous center of curvature" motion model

    :param state:
    :param noise:
    :param v: Speed
    :param w: Angular velocity
    :return:
    """
    Rv, Rw = noise

    # Corrupt the inputs with noise
    vc = v + v*Rv*dt
    wc = w + w*Rv*dt

    # Find the displacement using corrupted inputs
    next_state = state + get_displacement(state[2],vc,wc,dt)
    return next_state

def measurement(state, noise):
    """
    Convert the state into a measurement and corrupt it with noise
    The measurement consists of a bearing and range measurement--
    but the range measurement gets worse the further away you are

    :param state:
    :param noise:
    :return:
    """
    xy = state[:2]
    range = norm(xy)
    bearing = np.arctan2(xy[1],xy[0])
    Rr,Rb = noise

    # Corrupt measurements with non-additive noise
    range_c = max(0,range + Rr*range**2)
    bearing_c = bearing + Rb*range**2

    return np.array([range_c, bearing_c])


"""
A simple UKF application

Model a unicycle-model robot
The measurement is the range and bearing, 
but the noise increases with the square of distance
"""

# Process noise covariance
RV = np.diag([1e-3, 1e-4])

# Measurement noise covariance
RN = np.diag([0.01, 0.001])

ukf = ukflib.UnscentedKalmanFilter(state_size=3,
                                   process_noise=RV,
                                   measurement_noise=RN)



# Generate a realistic trajectory using numerical integration
# Suppose the robot smoothly changes its speed and angular velocity
# Make up some speeds and angular velocities and interpolate between them
speeds =   [0, 2.0,  1.0, 0.5, 0, 0]
ang_vels = [0,  90,  -180,  -360, 0, 0]
ang_vels = np.radians(ang_vels)
times =    [0,1,2,3,4,5]
V = scipy.interpolate.interp1d(times, speeds)
W = scipy.interpolate.interp1d(times, ang_vels)
f = lambda x,t: np.array([V(t)*cos(x[2]), V(t)*sin(x[2]), W(t)])

tf = 4
dt = 0.01
T = np.arange(0,tf,dt)

# The ground truth states
X = scipy.integrate.odeint(f, [0,0,0], T)

for i, truth in enumerate(X):
    t = T[i]
    input_v = V(t)
    input_w = W(t)
    ukf.predict(predict_icc, input_v, input_w, dt)

    range = norm(truth)
    bearing = atan2(truth[1], truth[0])

    # Convert variances to standard deviation
    r_std = sqrt(RN[0,0])
    b_std = sqrt(RN[1,1])

    # Fuzz it up with noise
    range += random.gauss(0, r_std)*range**2
    bearing += random.gauss(0, b_std)*range**2

    z = np.array([range, bearing])
    ukf.update(measurement, z)


plt.plot(X[:,0],X[:,1])
plt.show()




