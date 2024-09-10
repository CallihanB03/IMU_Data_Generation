import torch
from torch.distributions import Distribution
import random

class Latent_Distribution(Distribution):
    def __init__(self, z_loc, z_scale):
        assert z_loc.shape == z_scale.shape, "z_loc and z_scale must be the same size"

        super(Latent_Distribution, self).__init__(validate_args=False)
        self.z_loc = z_loc.float()
        self.z_scale = z_scale.float()

        self.epsilon = torch.randn_like(self.z_loc)

    def __draw_sample(self):
        return self.epsilon * self.z_scale + self.z_loc

    """Draws samples from latent distribution; only works for latent dimensionality of 2"""
    def sample(self, shape=None):
        if not shape:
            shape = self.z_loc.shape

        return self.epsilon * self.z_scale + self.z_loc

        # latent_sample = torch.zeros(shape)
        # rows, cols = shape
        # for row in range(rows):
        #     for col in range(cols):
        #         latent_sample[row][col] = self.__draw_sample()
        # return latent_sample
    

if __name__ == "__main__":
    z_loc = torch.tensor([0, 0])
    z_loc = torch.unsqueeze(z_loc, dim=0)

    z_scale = torch.tensor([1,1])
    z_scale = torch.unsqueeze(z_scale, dim=0)

    
    latent_dist = Latent_Distribution(z_loc=z_loc, z_scale=z_scale)
    print(latent_dist.sample())
