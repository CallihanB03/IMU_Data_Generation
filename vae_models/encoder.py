import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.loc = nn.Linear(self.hidden_size, self.latent_size)
        self.scale = nn.Linear(self.hidden_size, self.latent_size)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        z_loc = self.loc(x)
        z_scale = self.elu(self.scale(x))
        return z_loc, z_scale



class Sampling_Layer(nn.Module):
    def __init__(self):
        super(Sampling_Layer, self).__init__()


    def forward(self, inputs):
        z_loc, z_scale = inputs
        batch = z_loc.shape[0]
        dim = z_scale.shape[1]
        z_loc, z_scale = torch.mean(z_loc, dim=0), torch.mean(z_scale, dim=0)
        epsilon = torch.randn(batch, dim)
        return z_loc + torch.exp(0.5 * z_loc) * epsilon

if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import df_to_tensor, tensor_train_test_split
    data = load_participant_data()
    data = data[data.columns[:-3]] # Removing string arguments

    data = df_to_tensor(data)

    train, test = tensor_train_test_split(data, test_size=0.2)

    input_size = train.shape[1]

    encoder = Encoder(input_size=input_size, hidden_size=32, latent_size=2)
    sampling = Sampling_Layer()

    z_loc, z_scale = encoder(train)

    print(f"z_loc = {z_loc}")
    print(f"z_scale = {z_scale}")

    z_space = (z_loc, z_scale)

    z_sample = sampling(z_space)

    print(f"z_sample = {z_sample}")




