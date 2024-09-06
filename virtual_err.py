from physics_python.kinematics_appr import calculate_time, calculate_velocity, sample_acceleration, calculate_time_change
from physics_python.virtual_error import virtual_error
from util.participant_data import load_participant_data
import pandas as pd
from torch import tensor, mean, std
import json

def calculate_vir_err_distributions(save=False, save_path=None):
    df = load_participant_data()

    participants = [int(participant[1:]) for participant in pd.unique(df["participant"])]
    participants.sort()

    directions = ['x', 'y', 'z']
    virtual_error_dists = {}

    for participant in participants:
        virtual_error_dists[participant] = {}
        delta_t = calculate_time_change(participant)
        for direction in directions:
            loc, scale = __create_virtual_error_distribution(df, participant, direction, delta_t)
            virtual_error_dists[participant][direction] = tensor((loc, scale))

    if save and not save_path:
        save_path = "./data/virtual_error_distribution.json"
    
    if save:
        with open(save_path, 'w') as file:
            json.dump(virtual_error_dists, file)
    return virtual_error_dists
    
            
          


def __create_virtual_error_distribution(df, participant, direction, delta_t):
    errs = []
    for _ in range(10):
        acceleration = sample_acceleration(df, participant, direction)
        velocity = calculate_velocity(acceleration, delta_t)
        err = virtual_error(velocity, acceleration, delta_t)
        errs.append(err)

    errs = tensor(errs)

    return mean(errs).item(), std(errs).item()



if __name__ == "__main__":
    errs_dict = calculate_vir_err_distributions(save=False)
    for first_key in errs_dict:
        print(f"participant = {first_key}")
        for second_key in errs_dict[first_key]:
            print(f"    {errs_dict[first_key][second_key]}")
        print()