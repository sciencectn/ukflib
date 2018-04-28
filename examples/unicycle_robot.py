

import ukflib
import numpy as np
from math import sin,cos, atan2, sqrt, radians
from numpy.linalg import norm
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import random
from collections import defaultdict as ddict

"""
An example UKF application

We have a unicycle-model robot (e.g. tank treads)
which moves on some random trajectory. Its inputs are speed
and angular velocity. 

We can only measure the range and bearing to the robot, 
so we have to randomly guess the inputs. 
"""


def rotate(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]])

def get_displacement(ts, v, omega, dt):
    w = np.zeros(3)
    if abs(omega) < 1e-5:
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
    Rv, Rw, Ra = noise

    # Corrupt the inputs with noise
    vc = v + v*Rv*dt
    wc = w + w*Rw*dt

    # Find the displacement using corrupted inputs
    next_state = state + get_displacement(state[2],vc,wc,dt)

    # Add a third bit of uncertainty to just the angle
    # This prevents the orientation estimate from getting too confident
    # when the robot is not turning
    next_state[2] += Ra*dt
    return next_state

def clip(value, low, high):
    return max(min(value, high), low)

def predict_random_inputs(state, noise, dt, vmin, vmax, wmin, wmax):
    v,w = state[3:] + noise*dt
    v = clip(v,vmin,vmax)
    w = clip(w,wmin,wmax)
    next_state = state
    next_state[:3] += get_displacement(state[2],v,w,dt)
    # next_state[:3] += noise[:3]*dt
    next_state[3] = v
    next_state[4] = w
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



# Process noise covariance
# In this case, just how fast v and omega are changing
RV = np.diag([0.5,0.5])*5

# Measurement noise covariance
# One is the actual statistics of the noise,
# the other is what we tell the UKF. If the UKF gets too confident,
# we can increase what it thinks the measurement noise is.
measurement_noise_actual = np.diag([0.01, 0.001])
measurement_noise_ukf = np.diag([0.01, 0.001])*10

# Create the UKF object
# The states are: x,y,heading,v (speed input), omega (turn speed input)
# We are including v and omega as states because the UKF is going to guess them
# Normally, v and omega would not be states because the robot simple tells us its inputs.
ukf = ukflib.UnscentedKalmanFilter(state_size=5,
                                   process_noise=RV,
                                   measurement_noise=measurement_noise_ukf,
                                   init_covariance=np.eye(5)*1e-2,
                                   init_state=[1,0,0,0.5,0],
                                   angle_mask=[0,0,1,0,0],
                                   alpha=0.5)


# Generate a realistic trajectory using numerical integration
# Suppose the robot smoothly changes its speed and angular velocity
# Make up some speeds and angular velocities and interpolate between them

# The min/max inputs that the robot receives
vmin = 0.5
vmax = 2.0
wmin = -0.5
wmax = 0.5

# Generate random inputs at certain time intervals,
# but smoothly transition between the random inputs
# This simulates the robot smoothly accelerating.
num_points = 20
speeds = np.random.uniform(vmin,vmax,num_points)
speeds[0] = 0.5
ang_vels = np.random.uniform(wmin,wmax,num_points)
ang_vels[0] = 0
times = np.arange(0,num_points)*3.0  # Stretch out the times a bit
V = scipy.interpolate.interp1d(times, speeds)
W = scipy.interpolate.interp1d(times, ang_vels)

# Right hand side of state-space equations
f = lambda x,t: np.array([V(t)*cos(x[2]), V(t)*sin(x[2]), W(t)])

tf = np.max(times)-2.0
dt = 0.05
T = np.arange(0,tf,dt)

# Calculate the ground truth states using numerical integration
X_true = scipy.integrate.odeint(f, [1, 0, 0], T)

# Make a dict to record some data throughout the UKF run
d = ddict(lambda: [])
d["X_true"] = X_true

# Run a prediction and measurement on the UKF for each timestep
for i, truth in enumerate(X_true):
    t = T[i]

    # Predict the next state and covariance using our prediction callback
    ukf.predict(predict_random_inputs, dt, vmin, vmax, wmin, wmax)

    # Generate some fake inputs with noise
    range = norm(truth)
    bearing = atan2(truth[1], truth[0])
    z = np.array([range, bearing])
    z += np.random.multivariate_normal(np.zeros(2), measurement_noise_actual)

    # Run the update
    ukf.update(measurement, z)

    # Collect some data about the UKF to plot later
    d["X_ukf"].append(ukf.state)
    xy_cov = ukf.covariance[:2,:2]
    e,_ = np.linalg.eig(xy_cov)
    # An estimate of the average uncertainty in all directions
    est_std = np.mean(np.sqrt(e))
    d["X_cov"].append(est_std)

for k,v in d.items():
    d[k] = np.array(v)

plt.subplot(1,4,1)
plt.plot(d["X_true"][:, 0], d["X_true"][:, 1],label="Truth")
plt.plot(d["X_ukf"][:,0],d["X_ukf"][:,1],label="UKF")
plt.title("XY")
plt.legend()

plt.subplot(1,4,2)
plt.plot(T, d["X_cov"])
plt.title("Estimated error (average std dev)")

plt.subplot(1,4,3)
theta_true = d["X_true"][:,2]
theta_ukf = d["X_ukf"][:,2]
a_err = np.degrees(ukflib.angular_fix(theta_true - theta_ukf))
plt.plot(T,a_err)
plt.title("Orientation error (degrees)")

plt.subplot(1,4,4)
error = norm(d["X_true"][:,:2] - d["X_ukf"][:,:2], axis=1)
plt.plot(T,error)
plt.title("XY Error")

plt.figure()
plt.subplot(1,2,1)
plt.plot(T,V(T),label="V true")
plt.plot(T,d["X_ukf"][:,3],label="V ukf")
plt.legend()

plt.subplot(1,2,2)
plt.plot(T,W(T),label="$\omega$ true")
plt.plot(T,d["X_ukf"][:,4],label="$\omega$ ukf")
plt.legend()


plt.show()


