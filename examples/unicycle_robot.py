from __future__ import print_function
import ukflib
import numpy as np
from math import sin,cos, atan2
from numpy.linalg import norm
import scipy.interpolate
import scipy.integrate
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


def clip(value, val_range):
    return max(min(value, val_range[1]), val_range[0])

def predict_random_inputs(state, noise, dt, vrange, wrange):
    v,w=state[3:]
    next_state = state
    next_state[:3] += get_displacement(state[2],v,w,dt)
    next_state[3:] += noise[:3]*dt
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


def generate_random_trajectory_and_inputs(dt, vrange, wrange, num_points=20):
    """
    Generate a realistic trajectory using numerical integration
    Suppose the robot smoothly changes its speed and angular velocity
    Make up some speeds and angular velocities and interpolate between them

    This simulates the robot smoothly accelerating.
    :param dt:  How often to sample the true state
    :param vrange:  Range of allowable speed inputs  (min,max)
    :param wrange:  Range of allowable angular velocity inputs  (min,max)
    :param num_points:
    :return (T, X_true, V, W): The array of times, the true state, and the input functions
                which return the input at any time t: V(t), W(t)
    """

    speeds = np.random.uniform(vrange[0], vrange[1], num_points)
    speeds[0] = 0.5
    ang_vels = np.random.uniform(wrange[0], wrange[1], num_points)
    ang_vels[0] = 0
    times = np.arange(0, num_points) * 3.0  # Stretch out the times a bit
    V = scipy.interpolate.interp1d(times, speeds)
    W = scipy.interpolate.interp1d(times, ang_vels)

    # Right hand side of state-space equations
    f = lambda x, t: np.array([V(t) * cos(x[2]), V(t) * sin(x[2]), W(t)])

    tf = np.max(times) - 2.0
    T = np.arange(0, tf, dt)

    # Calculate the ground truth states using numerical integration
    X_true = scipy.integrate.odeint(f, [1, 0, 0], T)
    return T,X_true,V,W


def run_ukf(times,
            dt,
            vrange,
            wrange,
            true_states,
            ukf_measurement_noise,
            ukf_process_noise,
            alpha,
            V=None,
            W=None,
            plot=False):
    """
    Generate fake measurements from the true states
    Feed these fake inputs into the UKF and return the overall accuracy
    :param times: Time values at which true states are sampled
    :param dt:  Timestep between each time
    :param vrange:  Tuple, range of speed input used (min,max)
    :param wrange:  Tuple, range of angular velocity input (min,max)
    :param true_states: True state of the robot
    :param ukf_measurement_noise: The measurement noise variances to feed to the UKF.
                                Not the same as the real measurement noise (tunable)
    :param ukf_process_noise:   Process noise variances (tunable)
    :param alpha:   UKF parameter (tunable)
    :param V:       Function: given t, return the speed
    :param W:       Function: given t, return angular velocity input
    :param plot:    Plot the results
    :return (avg_error, pctl_error90): the average and 90th
        percentile XY errors
    """
    actual_measurement_noise = np.diag([0.01, 0.001])

    # Create the UKF object
    # The states are: x,y,heading,v (speed input), omega (turn speed input)
    # We are including v and omega as states because the UKF is going to guess them
    # Normally, v and omega would not be states because the robot simple tells us its inputs.
    ukf = ukflib.UnscentedKalmanFilter(state_size=5,
                                       process_noise=ukf_process_noise,
                                       measurement_noise=ukf_measurement_noise,
                                       init_covariance=np.eye(5) * 1e-2,
                                       init_state=[1, 0, 0, 0.5, 0],
                                       angle_mask=[0, 0, 1, 0, 0],
                                       alpha=alpha,
                                       repair_covariance=True)

    # Make a dict to record some data throughout the UKF run
    d = ddict(lambda: [])
    d["X_true"] = true_states

    # Run a prediction and measurement on the UKF for each time step
    for i, truth in enumerate(true_states):
        # Predict the next state and covariance using our prediction callback
        ukf.predict(predict_random_inputs, dt, vrange, wrange)

        # Generate some fake inputs with noise
        range = norm(truth)
        bearing = atan2(truth[1], truth[0])
        z = np.array([range, bearing])
        z += np.random.multivariate_normal(np.zeros(2), actual_measurement_noise)

        # Run the update
        ukf.update(measurement, z)

        # Collect some data about the UKF to plot later
        d["X_ukf"].append(ukf.state)
        xy_cov = ukf.covariance[:2, :2]
        e, _ = np.linalg.eig(xy_cov)
        # An estimate of the average uncertainty in all directions
        est_std = np.mean(np.sqrt(e))
        d["X_cov"].append(est_std)

    for k, v in d.items():
        d[k] = np.array(v)
    if plot:
        T = times
        plt.figure(figsize=(13,9))
        plt.subplot(2, 3, 1)
        plt.plot(d["X_true"][:, 0], d["X_true"][:, 1], label="Truth")
        plt.plot(d["X_ukf"][:, 0], d["X_ukf"][:, 1], label="UKF")
        plt.scatter([1],[0],s=20.0,label="Start")
        plt.title("XY")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(T, d["X_cov"])
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("Estimated error (average std dev)")

        plt.subplot(2, 3, 6)
        theta_true = d["X_true"][:, 2]
        theta_ukf = d["X_ukf"][:, 2]
        a_err = np.degrees(ukflib.angular_fix(theta_true - theta_ukf))
        plt.plot(T, a_err)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (degrees)")
        plt.title("Orientation error")

        plt.subplot(2, 3, 3)
        error = norm(d["X_true"][:, :2] - d["X_ukf"][:, :2], axis=1)
        plt.plot(T, error)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("XY Error")

        # plt.figure()
        plt.subplot(2, 3, 4)
        plt.plot(T, V(T), label="V true")
        plt.plot(T, d["X_ukf"][:, 3], label="V ukf")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Estimated input v")
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(T, W(T), label="$\omega$ true")
        plt.plot(T, d["X_ukf"][:, 4], label="$\omega$ ukf")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular speed (rads/s)")
        plt.title("Estimated input $\omega$")
        plt.legend()

        plt.tight_layout(pad=0.1)
        plt.savefig("unicycle_plot.pdf")

        plt.show()
    true_xy = d["X_true"][:,:2]
    est_xy = d["X_ukf"][:,:2]
    error_xy = norm(true_xy - est_xy, axis=1)
    error_angle = ukflib.angular_fix(d["X_true"][:,2] - d["X_ukf"][:,2])
    error = np.array([error_xy, error_angle]).T
    # Weight it so 1 meter of distance error is x degrees of angular error
    weights = np.array([np.radians(90), 1.0])
    error = error.dot(weights) / sum(weights)
    avg_error = np.mean(error)
    pctl_error90 = np.percentile(error, 90)
    return avg_error, pctl_error90

if __name__=="__main__":
    import matplotlib.pyplot as plt
    process_noise = np.diag([7.0,7.0])
    measurement_noise_ukf = np.diag([0.9, 0.005])

    # The min/max inputs that the robot receives
    vrange = (0.5,2.0)      # min/max forward speed
    wrange = (-0.5,0.5)     # min/max angular velocity

    dt = 0.05
    times, true_state, input_v, input_w = generate_random_trajectory_and_inputs(dt, vrange, wrange)
    avg,pctl=run_ukf(times,
            dt,
            vrange,
            wrange,
            true_state,
            measurement_noise_ukf,
            process_noise,
            alpha=0.357,
            V=input_v,
            W=input_w,
            plot=True)
    print("Average error={0}, 90th percentile error={1}".format(avg,pctl))

