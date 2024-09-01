#!/usr/bin/env python3
import torch

def calculate_time(data, participant):
    """
    sampling rate for IMU data was 55 Hz
    as is cited on page 6, section 4.1 in
    Detecting In-Person Conversations in Challenging 
    Real-World Environments with Smartwatch Audio and Motion Sensing
    """
    sampling_rate = 55
    participant_type_1 = (3, 8)
    participant_type_2 = (4, 6, 9, 11, 13, 16, 18, 20, 22, 24)
    participant_type_3 = (i for i in range(4, 25) if (i not in participant_type_1 and i not in participant_type_2))

    window_size_hash = {
        participant_type_1: 1485,
        participant_type_2: 1521,
        participant_type_3: 1649
    }

    for key in window_size_hash:
        if participant in key:
            window_size = window_size_hash[key]
            break 
    
    # Sampling period
    delta_t = 1 / sampling_rate

    window_time = delta_t * window_size

    participant_string = 'p' + str(participant)
    time = torch.arange(0, window_time * len(data[data.participant == participant_string]), window_time)
    return time, delta_t
  



def calculate_velocity(data, participant, direction):
    _, delta_t = calculate_time(data, participant)


    participant_str = 'p' + str(participant)
    acc_mean_str = f"{direction}_acc_mean"
    acc_std_str = f"{direction}_acc_mean"

    participant_data = data[data.participant == participant_str]
    acc_mean = torch.tensor(participant_data[acc_mean_str])
    acc_std = torch.tensor(participant_data[acc_std_str])

    # acceleration = epsilon * acc_std + acc_mean where epsilon ~ N(0, I)
    # approximating acceleration values over time
    epsilon_values = torch.randn_like(acc_mean)
    accelerations = epsilon_values * acc_std + acc_mean

    initial_velocity = 0 # began study by resting our wrist on the table
    velocity = initial_velocity + torch.cumsum(accelerations, dim=0) * delta_t
    return velocity


    


if __name__ == "__main__":
    # python -m physics_python.kinematics_appr
    from util.load_data import load_data
    
    df = load_data()
    participant = 3
    direction = 'x'
    time, delta_t = calculate_time(df, participant)
    print(f"time = {time}")
    print(f"delta_t = {delta_t}")

    velocity = calculate_velocity(df, participant, direction)
    print(f"velocity = {velocity}")