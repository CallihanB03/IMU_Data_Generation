import torch
import torch.nn as nn

def virtual_error(velocity_tensor, acceleration_tensor, time_window):
    pred_velocities = torch.zeros_like(velocity_tensor)
    for ind in range(velocity_tensor[:-1].size(dim=0)):
        pred_velocity = velocity_tensor[ind].item() + acceleration_tensor[ind].item() * time_window
        pred_velocities[ind+1] = pred_velocity

    print(f"velocity_tensor = {velocity_tensor}")
    print(f"pred_velocities = {pred_velocities}")
    
    return nn.MSELoss()(velocity_tensor, pred_velocities)

    








if __name__ == "__main__":
    from physics_python.kinematics_appr import calculate_time, calculate_velocity, sample_acceleraton
    from util.load_data import load_data


    df = load_data()
    _, delta_t = calculate_time(df, participant=3)
    velocity = calculate_velocity(df, participant=3, direction='x')
    acceleration = sample_acceleraton(df, participant=3, direction='x')
    v_err = virtual_error(velocity, acceleration, delta_t)
    
    print(f"virtual error = {v_err}")

