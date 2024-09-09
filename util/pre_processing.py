import torch
import random

def df_to_tensor(df):
    cols = len(df.columns)
    rows = len(df)
    tensor_df = torch.zeros(rows, cols)
    for col_index, col in enumerate(df.columns):
        for row in range(rows):
            tensor_df[row, col_index] = df[col][row]
    return tensor_df

def tensor_remove(tensor, remove_value):
    for index, value in enumerate(tensor):
        if value.item() == remove_value:
            tensor = torch.cat(tensor[:index], tensor[index+1:])
        return tensor
    print(f"{remove_value} was not found in tensor")
    return tensor


def tensor_train_test_split(tensor, test_size):
    assert (0 <= test_size <= 1), "test_size must be in the range [0, 1]"
    train_size = 1 - test_size
    df_num_rows = tensor.shape[0]

    train_num_rows = int(train_size * df_num_rows)
    train_indices = []
    test_indices = []

    total_indices = [i for i in range(df_num_rows)]

    for _ in range(train_num_rows):
        random_index = random.choice(total_indices)
        total_indices.remove(random_index)
        train_indices.append(random_index)

    test_indices = total_indices


    train_tensor = tensor[train_indices]
    test_tensor = tensor[test_indices]

    return train_tensor, test_tensor


if __name__ == "__main__":
    from participant_data import load_participant_data
    df = load_participant_data()
    df = df[df.columns[:-3]] # removing non-Float values from df
    df = df_to_tensor(df)

    train, test = tensor_train_test_split(df, test_size=0.2)

    print(f"train tensor shape  = {train.shape}")
    print()
    print(f"test tensor shape = {test.shape}")
