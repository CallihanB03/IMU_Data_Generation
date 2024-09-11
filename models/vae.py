from encoder import Encoder, Sampling_Layer
from decoder import Decoder
from distributions.distribution import Transform_Distribution
import torch
import torch.nn as nn



class VAE(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size, transformation_hidden_size=None, standard_encoder=True):
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.transformation_hidden_size = transformation_hidden_size
        self.standard_encoder = standard_encoder

        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, latent_size=self.latent_size)
        self.decoder = Decoder(latent_size=self.latent_size, hidden_size=self.hidden_size, output_size=self.data_size)
        self.sampling_layer = Sampling_Layer()

        self.transformation_module = Transform_Distribution(dist_input_size=self.latent_size, hidden_size=transformation_hidden_size)

        
    def forward(self, x):
        dist_params = self.encoder(x)
        latent_space = self.sampling_layer(dist_params)
        transformed_dist = self.transformation_module(latent_space)
        reconstruction = self.decoder(transformed_dist)
        return reconstruction
    

if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import df_to_tensor, tensor_train_test_split
    data = load_participant_data()
    data = data[data.columns[:-3]] # Removing string arguments

    data = df_to_tensor(data)

    train, test = tensor_train_test_split(data, test_size=0.2)

    input_size = train.shape[1]




    vae = VAE(data_size=input_size, hidden_size=64, latent_size=32)

    reconstruction = vae(train)

    print(reconstruction)

