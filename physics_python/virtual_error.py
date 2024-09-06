import torch
import torch.nn as nn

def virtual_error(velocity_tensor, acceleration_tensor, time_window):
    pred_velocities = torch.zeros_like(velocity_tensor)
    for ind in range(velocity_tensor[:-1].size(dim=0)):
        pred_velocity = velocity_tensor[ind].item() + acceleration_tensor[ind].item() * time_window
        pred_velocities[ind+1] = pred_velocity
        
    return nn.MSELoss()(velocity_tensor, pred_velocities)

    








if __name__ == "__main__":
    from physics_python.kinematics_appr import calculate_time, calculate_velocity, sample_acceleration, calculate_time_change
    from util.participant_data import load_participant_data
    participant=3
    direction='x'


    df = load_participant_data()
    delta_t = calculate_time_change(participant)
    acceleration = sample_acceleration(df, participant, direction='x')
    velocity = calculate_velocity(acceleration, delta_t)


    v_err = virtual_error(velocity, acceleration, delta_t)
    
    print(f"virtual error = {v_err}")

