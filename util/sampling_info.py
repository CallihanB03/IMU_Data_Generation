import json

def download_sampling_info_data():
    participant_type_1 = (3, 8)
    participant_type_2 = (4, 6, 9, 11, 13, 16, 18, 20, 22, 24)
    participant_type_3 = (i for i in range(4, 25) if (i not in participant_type_1 and i not in participant_type_2))

    window_size_hash = {
        "sampling_rate": 55,
        "participant_type_1": 1485,
        "participant_type_2": 1521,
        "participant_type_3": 1649
    }

    save_path = "./data/sampling_info.json"
    with open(save_path, 'w') as file:
        json.dump(window_size_hash, file)

    print(f"JSON file containing information on lab sampling rate and participant sampling window sizes has been saved to {save_path}")
    return None



def load_sampling_info_data(save_path=None):
    if not save_path:
        save_path = "./data/sampling_info.json"
    with open(save_path, 'r') as sampling_info_data:
        data = json.load(sampling_info_data)
    return data


if __name__ == "__main__":
    download_sampling_info_data()

    sampling_info = load_sampling_info_data()

    for key in sampling_info.keys():
        print(f"{key}: {sampling_info[key]}")


    