
import ukflib
import numpy as np
import random
import matplotlib.pyplot as plt
import math

"""
The state is the angle of something spinning around an axis, 
like a ball on a string. We can only measure its x position. 
"""

# Propagation function callback: given a state, return the next state
def predict(state,noise,dt):
    return state + 1.0*dt + noise*dt


# Measurement function callback: given a state and noise, return the measurement
def measurement(state, noise):
    return np.cos(state + noise)


Rv = 1e-6 # Process noise variance
Rn = np.atleast_2d(0.1)  # Measurement noise variance


ukf = ukflib.UnscentedKalmanFilter(1,
                                   Rv,
                                   Rn,
                                   init_covariance=0.1,
                                   init_state=0,
                                   angle_mask=[1])
dt = 0.01
tf = 6.0     # Final time
T = np.arange(0,tf,dt)   # All time values
truth = []
x_ukf = []
true_state = 0

# Iterate through each timestep, create a fake measurement,
# then update the UKF
for t in T:
    ukf.predict(predict, dt)
    true_state += 1.0*dt + random.gauss(0,(Rv*dt)**0.5)
    true_state = ukflib.angular_fix(true_state)
    truth.append(true_state)

    noise = random.gauss(0,Rn[0]**0.5)
    z = math.cos(true_state + noise)
    ukf.update(measurement, z)
    x_ukf.append(ukf.state[0])
x_ukf = np.array(x_ukf)

truth = np.array(truth)



plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(T,truth,label="True state")
plt.plot(T,x_ukf,label="UKF state")
plt.xlabel("Time (s)")
plt.ylabel("State (radians)")
plt.title("States")
plt.legend()


error = np.degrees(np.abs(ukflib.angular_fix(x_ukf - truth)))
plt.subplot(1,2,2)
plt.plot(T,error)
plt.title("Estimation error")
plt.xlabel("Time (s)")
plt.ylabel("Error (degrees)")

plt.savefig("angle_example.pdf")

plt.show()

