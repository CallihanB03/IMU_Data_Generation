import torch

def df_to_tensor(df):
    cols = len(df.columns)
    rows = len(df)
    tensor_df = torch.zeros(rows, cols)
    for col_index, col in enumerate(df.columns):
        for row in range(rows):
            tensor_df[row, col_index] = df[col][row]
    return tensor_df

if __name__ == "__main__":
    from participant_data import load_participant_data
    df = load_participant_data()
    df = df[df.columns[:-3]] # removing non-Float values from df
    df = df_to_tensor(df)
