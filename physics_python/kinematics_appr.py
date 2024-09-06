#!/usr/bin/env python3
import torch
from util.sampling_info import load_sampling_info_data

"""
The data is organized in a pandas dataframe with each row corresponding to a time segment.

(*) Approximated a time tensor for the data by:
1) Calculating sampling_period 
    sampling_period = 1 / sampling frequency = 1 / 55 Hz = 0.01818...

2) Multiplying sampling_period by sampling window to approximate 
the length of the sampling window in seconds. 
Suppose the sampling window = 1485, then
    sampling_len_in_seconds = sampling_period * sampling_window = 27 seconds

Then, in this example, each time segment is assumed to be approximately 27 seconds.

Each segment has a acceleration mean and acceleration standard deviation for
each of the x, y, and z directions.

The acceleration tensor for the data is given as a tensor where each element is drawn from a
Normal distribution with mean=acceleration mean and standard deviation=acceleration standard deviation.

(**) Approximated a velocity tensor for the data by using the definition of velocity.
v(t) = ∫a(t) + v(0) where v(t) and a(t) are functions of velocity and acceleration
wrt time respectively.

v(0) is assumed to be 0 m/s (I believe all studies began with 
participants resting their hands on a table)

"""
def calculate_time_change(participant):
    sampling_info = load_sampling_info_data()


    sampling_rate = sampling_info["sampling_rate"]
    sampling_period = 1 / sampling_rate

    participant_types = ["participant_type_1", "participant_type_2", "participant_type_3"]

    participant_type_num = 0
    for participant_type in participant_types:
        participant_type_num += 1
        if participant in sampling_info[participant_type]:
            window_size = sampling_info["windows"][str(participant_type_num)]
            break

    delta_t = sampling_period * window_size

    return delta_t


def calculate_time(data, participant, delta_t):
    
    """
    sampling rate for IMU data was 55 Hz
    as is cited on page 6, section 4.1 in
    Detecting In-Person Conversations in Challenging 
    Real-World Environments with Smartwatch Audio and Motion Sensing
    """

    participant_string = 'p' + str(participant)
    total_time = delta_t * len(data[data.participant == participant_string])
    time = torch.arange(0, total_time, delta_t)
    return time
  

def sample_acceleration(data, participant, direction):
    participant_str = 'p' + str(participant)
    acc_mean_str = f"{direction}_acc_mean"
    acc_std_str = f"{direction}_acc_mean"

    participant_data = data[data.participant == participant_str]
    acc_mean = torch.tensor(participant_data[acc_mean_str].to_numpy())
    acc_std = torch.tensor(participant_data[acc_std_str].to_numpy())


    # acceleration = epsilon * acc_std + acc_mean where epsilon ~ N(0, I)
    # approximating acceleration values over time
    epsilon_values = torch.randn_like(acc_mean)
    accelerations = epsilon_values * acc_std + acc_mean
    return accelerations
    


def calculate_velocity(accelerations, delta_t):
    initial_velocity = 0 # began study by resting our wrist on the table
    velocity = initial_velocity + torch.cumsum(accelerations, dim=0) * delta_t
    return velocity

    


if __name__ == "__main__":
    # python -m physics_python.kinematics_appr
    from util.participant_data import load_participant_data
    
    df = load_participant_data()
    participant = 3
    direction = 'x'


    delta_t = calculate_time_change(participant)
    time = calculate_time(df, participant, delta_t)
    print(f"time = {time}")
    print(f"delta_t = {delta_t}")

    acceleration = sample_acceleration(df, participant, direction)
    print(f"acceleration = {acceleration}")

    velocity = calculate_velocity(acceleration, delta_t)
    print(f"velocity = {velocity}")