import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc2 = nn.Linear (self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    


if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import df_to_tensor, tensor_train_test_split
    from models.standard_encoder import Encoder, Sampling_Layer
    data = load_participant_data()
    data = data[data.columns[:-3]] # Removing string arguments

    data = df_to_tensor(data)

    train, test = tensor_train_test_split(data, test_size=0.2)

    input_size = train.shape[1]

    encoder = Encoder(input_size=input_size, hidden_size=32, latent_size=2)
    sampling = Sampling_Layer()
    decoder = Decoder(output_size=input_size, hidden_size=32, latent_size=2)

    z_loc, z_scale = encoder(train)

    z_space = (z_loc, z_scale)

    z_sample = sampling(z_space)

    reconstruction = decoder(z_sample)

    print(f"z_sample = {z_sample}")

    print(f"Decoder output = {reconstruction}")
    print(f"Decoder output shape = {reconstruction.shape}")



    