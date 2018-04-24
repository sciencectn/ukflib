

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
    wc = w + w*Rw*dt

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
    range_c = max(0,range + Rr)
    bearing_c = bearing + Rb

    return np.array([range_c, bearing_c])


"""
A simple UKF application

Model a unicycle-model robot
The measurement is the range and bearing, 
but the noise increases with the square of distance
"""

# Process noise covariance
# RV = np.diag([1e-3, 1e-4])
RV = np.diag([0.5,0.5])

# Measurement noise covariance
RN = np.diag([0.01, 0.01])

ukf = ukflib.UnscentedKalmanFilter(state_size=3,
                                   process_noise=RV,
                                   measurement_noise=RN,
                                   init_covariance=np.eye(3)*0.01,
                                   angle_mask=[0,0,1])


# Generate a realistic trajectory using numerical integration
# Suppose the robot smoothly changes its speed and angular velocity
# Make up some speeds and angular velocities and interpolate between them

true_inputs = np.array([
    (0,   0,    0),
    (0.75, 90,   1),
    (1.0, -180, 2),
    (0.5, -200, 3),
    (0,  0,     4),
    (0,  0,     5)
])
speeds =   true_inputs[:,0]
ang_vels = np.radians(true_inputs[:,1])
times = true_inputs[:,2]
V = scipy.interpolate.interp1d(times, speeds)
W = scipy.interpolate.interp1d(times, ang_vels)

# Right hand side of state-space equations
f = lambda x,t: np.array([V(t)*cos(x[2]), V(t)*sin(x[2]), W(t)])

tf = 4
dt = 0.01
T = np.arange(0,tf,dt)

# The ground truth states
X_true = scipy.integrate.odeint(f, [0, 0, 0], T)
X_ukf = []
P_ukf = []


for i, truth in enumerate(X_true):
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
    range += random.gauss(0, r_std)
    bearing += random.gauss(0, b_std)

    z = np.array([range, bearing])
    ukf.update(measurement, z)

    X_ukf.append(ukf.state)
    P_ukf.append(ukf.covariance)

X_ukf = np.array(X_ukf)

pstats = []
for P in P_ukf:
    e,_ = np.linalg.eig(P)
    pstats.append(norm(e))
pstats = np.array(pstats)

plt.figure(1)
plt.plot(X_true[:, 0], X_true[:, 1],label="Truth")
plt.plot(X_ukf[:,0],X_ukf[:,1],label="UKF")
plt.title("XY")
plt.legend()

plt.figure(2)
plt.plot(T,X_true[:,2])
plt.plot(T,X_ukf[:,2])
plt.title("Orientation")

plt.figure(3)
plt.plot(T, pstats)
plt.title("Covariance")

plt.show()


