import torch
import random

def df_to_tensor(df):
    rows = len(df)
    if len(df.shape) == 1:
        cols = 1
        tensor_df = torch.zeros(rows)
        for row in range(rows):
            tensor_df[row] = df[row]
        return tensor_df
    cols = len(df.columns)
    tensor_df = torch.zeros(rows, cols)
    for col_index, col in enumerate(df.columns):
        for row in range(rows):
            tensor_df[row, col_index] = df[col][row]
    return tensor_df



def tensor_train_test_split(tensor_x, tensor_y, test_size):
    assert (0 <= test_size <= 1), "test_size must be in the range [0, 1]"
    train_size = 1 - test_size
    df_num_rows = tensor_x.shape[0]

    train_num_rows = int(train_size * df_num_rows)
    train_indices = []
    test_indices = []

    total_indices = [i for i in range(df_num_rows)]

    for _ in range(train_num_rows):
        random_index = random.choice(total_indices)
        total_indices.remove(random_index)
        train_indices.append(random_index)

    test_indices = total_indices


    train_tensor_x = tensor_x[train_indices]
    train_tensor_y = tensor_y[train_indices]


    test_tensor_x = tensor_x[test_indices]
    test_tensor_y = tensor_y[test_indices]

    return train_tensor_x, train_tensor_y, test_tensor_x, test_tensor_y


if __name__ == "__main__":
    from participant_data import load_participant_data
    df = load_participant_data()
    df = df[df.columns[:-2]] # removing non-Float values from df
    data, labels = df_to_tensor(df[df.columns[:-1]]), df_to_tensor(df[df.columns[-1]])

    x_train, y_train, x_test, y_test = tensor_train_test_split(data, labels, test_size=0.2)

    print(f"x_train tensor shape  = {x_train.shape}")
    print(f"y_train tensor shape = {y_train.shape}")
    print()
    print(f"x_test tensor shape = {x_test.shape}")
    print(f"y_test tensor shape = {y_test.shape}")
