
import ukflib
import numpy as np
import random
import matplotlib.pyplot as plt
import math

"""
The state is the angle of something spinning around an axis, 
like a ball on a string. We can only measure its x position. 
"""

def predict(state,noise,dt):
    return state + 1.0*dt + noise*dt

def measurement(state, noise):
    return np.cos(state)+noise


Rv = 0.001
Rn = np.atleast_2d(0.1)
ukf = ukflib.UnscentedKalmanFilter(1,
                                   Rv,
                                   Rn,
                                   init_covariance=0.1,
                                   init_state=0,
                                   angle_mask=[1])
dt = 0.01
tf = 6.0
T = np.arange(0,tf,dt)
truth = []
x_ukf = []
true_state = 0
for t in T:
    ukf.predict(predict, dt)
    true_state += 1.0*dt + random.gauss(0,(Rv*dt)**0.5)
    true_state = ukflib.angular_fix(true_state)
    truth.append(true_state)

    z = math.cos(true_state)
    noise = np.random.multivariate_normal(np.zeros(Rn.shape[0]),Rn)
    z += noise
    ukf.update(measurement, z)
    x_ukf.append(ukf.state[0])
x_ukf = np.array(x_ukf)

truth = np.array(truth)

error = np.abs(ukflib.angular_fix(x_ukf - truth))

plt.figure()
plt.plot(T,truth)
plt.plot(T,x_ukf)

plt.figure()
plt.plot(T,error)
plt.title("error")


plt.show()

