import torch
import torch.nn as nn
from torch.distributions import Distribution
import random
from vae_models.encoder import Encoder

class Latent_Distribution(Distribution):
    def __init__(self, z_loc, z_scale):
        assert z_loc.shape == z_scale.shape, "z_loc and z_scale must be the same size"

        super(Latent_Distribution, self).__init__(validate_args=False)
        self.z_loc = z_loc.float()
        self.z_scale = z_scale.float()

        self.epsilon = torch.randn_like(self.z_loc)

    """Draws samples from latent distribution; only works for latent dimensionality of 2"""
    def sample(self, shape=None):
        if not shape:
            shape = self.z_loc.shape

        return self.epsilon * self.z_scale + self.z_loc
    

class Transform_Distribution(nn.Module):
    def __init__(self, dist_input_size=100, hidden_size=None):
        super(Transform_Distribution, self).__init__()
        self.sample_size = dist_input_size
        if not hidden_size:
            self.hidden_size = self.sample_size

        self.fc1 = nn.Linear(self.sample_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.sample_size)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

if __name__ == "__main__":
    z_loc = torch.tensor([0, 0])
    z_loc = torch.unsqueeze(z_loc, dim=0)

    z_scale = torch.tensor([1,1])
    z_scale = torch.unsqueeze(z_scale, dim=0)

    transform_dist = Transform_Distribution(sample_size=z_scale.shape[1])
    
    latent_dist = Latent_Distribution(z_loc=z_loc, z_scale=z_scale)

    gaussian_sample = latent_dist.sample()

    print(f"sample for Gaussian distribution: {gaussian_sample}")

    new_dist = transform_dist(gaussian_sample)

    print(f"transformed distribution: {new_dist}")
