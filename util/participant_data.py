import json
import pandas as pd


def download_participant_data(save_path):
    # type_1 = w1485 and h1485
    participant_type_1 = [3, 8]

    # type_2 = w1521 and h1521
    participant_type_2 = [4, 6, 9, 11, 13, 16, 18, 20, 22, 24]

    # type_3 = w1649 and h1649
    participant_type_3 = [i for i in range(4, 25) if (i not in participant_type_1 and i not in participant_type_2)]

    participant = participant_type_1[0]

    data = pd.read_csv(f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1485_h1485.csv")

    for participant in participant_type_1[1:]:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1485_h1485.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)

    for participant in participant_type_2:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1521_h1521.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)

    for participant in participant_type_3:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1649_h1649.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)
    
    data.to_csv(save_path)


def load_participant_data():
    # type_1 = w1485 and h1485
    participant_type_1 = [3, 8]

    # type_2 = w1521 and h1521
    participant_type_2 = [4, 6, 9, 11, 13, 16, 18, 20, 22, 24]

    # type_3 = w1649 and h1649
    participant_type_3 = [i for i in range(4, 25) if (i not in participant_type_1 and i not in participant_type_2)]

    participant = participant_type_1[0]

    data = pd.read_csv(f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1485_h1485.csv")

    for participant in participant_type_1[1:]:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1485_h1485.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)

    for participant in participant_type_2:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1521_h1521.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)

    for participant in participant_type_3:
        data_path = f"/home/shared/azhang/social_signal_sensing/data/participants/p{participant}/imu_features/p{participant}_convo_imu_features_w1649_h1649.csv"
        new_data = pd.read_csv(data_path)
        data = pd.concat([data, new_data], axis=0, ignore_index=True)

    if "Unnamed: 0" in data.columns:
        del data["Unnamed: 0"]

    return data



if __name__ == "__main__":
    save_path = "./IMU_Data_Generation/data/participant_imu_data.csv"
    download_participant_data(save_path)