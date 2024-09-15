from vae_models.encoder import Encoder, Sampling_Layer
from vae_models.decoder import Decoder
import torch
import torch.nn as nn



class VAE(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    
        self.z_loc = torch.zeros(1, self.latent_size)
        self.z_scale = torch.ones(1, self.latent_size)

        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, latent_size=self.latent_size)
        self.decoder = Decoder(latent_size=self.latent_size, hidden_size=self.hidden_size, output_size=self.data_size)
        self.sampling_layer = Sampling_Layer()



    def forward(self, x):
        latent_dist_params = self.encoder(x)
        latent_sample = self.sampling_layer(latent_dist_params)
        reconstruction = self.decoder(latent_sample)
        return reconstruction
    

if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import df_to_tensor, tensor_train_test_split
    data = load_participant_data()
    data = data[data.columns[:-3]] # Removing string arguments

    data = df_to_tensor(data)

    train, test = tensor_train_test_split(data, test_size=0.2)

    input_size = train.shape[1]




    vae = VAE(data_size=input_size, hidden_size=32, latent_size=2)

    reconstruction = vae(train)

    print(f"reconstruction = {reconstruction}")
    print(f"reconstruction.shape = {reconstruction.shape}")

