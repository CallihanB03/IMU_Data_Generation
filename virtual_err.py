from physics_python.kinematics_appr import calculate_time, calculate_velocity, sample_acceleraton
from physics_python.virtual_error import virtual_error
from util.load_data import load_data
import pandas as pd
import json

def existing_virtual_error():
    df = load_data()

    participants = [int(participant[1:]) for participant in pd.unique(df["participant"])]
    participants.sort()

    directions = ['x', 'y', 'z']

    virtual_error_hash = {}
    for participant in participants:
        _, delta_t = calculate_time(df, participant)
        for direction in directions:
            errs = []
            velocity = calculate_velocity(df, participant, direction)
            acceleration = sample_acceleraton(df, participant, direction)
            vir_err = virtual_error(velocity, acceleration, delta_t)
            errs.append(vir_err)
        virtual_error_hash[participant] = errs

    save_path = "./data/true_virtual_error.json"
    with open(save_path, 'w') as file:
        json.dump(virtual_error_hash, file)
    


    
        




    pass



if __name__ == "__main__":
    existing_virtual_error()