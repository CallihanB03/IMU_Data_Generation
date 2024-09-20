# Addressing The Lack Of IMU Data For Sensor Based Human Activity Recognition Tasks

### Abstract
In the era of wearable devices, human activity recognition (HAR) has become an increasingly popular domain of machine learning. A notorious obstacle in training reliable HAR models is the limited availability of inertial motion unit (IMU) data available. Privacy concerns and difficulties associated with conducting formal studies make the collection of IMU data costly and time-consuming. To overcome these challenges, we explore the use of probabilistic programming and physics informed machine learning techniques to investigate the potential use of generative modeling in the training of HAR systems.


### Background
#### The Standard Variational Autoencoder
A Variational Autoencoder (VAE) is a generative model that combines principles from probabilistic programming and deep learning to learn a latent space associated with higher dimensional data. VAEs can generate new instances of data by sampling from the learned latent space distribution.

[VAE Image]


The VAE consists of two networks: the Encoder, $E$, and the decoder, $D$. The encoder learns a map from the input space, $\mathbf{X}$, to the latent space, $q_{\psi}(\mathbf{z}|\mathbf{x})$, while the decoder learns a map from the latent space to the the output space, $\mathbf{\hat{X}}$.
![image](./README_images/VAE%20Diagram.png)
$$
E: \mathbf{X} \longrightarrow q_{\psi}(\mathbf{z}|\mathbf{x})
$$

$$
D: q_{\psi}(\mathbf{z}|\mathbf{x}) \longrightarrow \mathbf{\hat{X}}
$$

The objective of the VAE is defined as:

$$
\argmin_{\mathbf{\mu_{z}}, \mathbf{\Sigma_{z}}, \mathbf{\hat{x}}}\sum_{\mathbf{x}}\|\mathbf{x} - \mathbf{\hat{x}}\|_{2}^{2} + \beta KL(p_{\theta}(z)\|q_{\psi}(\mathbf{z}|\mathbf{x}))
$$

where $p_{\theta}(\mathbf{x})$ is a simple Gaussian, $q_{\psi}(\mathbf{z}|\mathbf{x})$ is the latent space learned by the VAE, and $\beta$ is a hyper-parameter to be tuned during training. The first term in the loss, referred to the reconstruction term, penalizes large differences in the model's input data and output data while the second term encourages a Gaussian latent space.

#### Transformation VEA

While standard 
VAEs have shown that they are able to generate realistic data, they do have a significant drawback. That is, they assume the latent space distribution to be Gaussian; however, this may not always be a good assumption since it is possible that the latent space may have a much more complex distribution. 

To address this, we will be implementing transformations to the latent space of the standard VAE to learn a more complex latent space that can better approximates the true latent space. These transformations are learned via a network that trains alongside the VAE.

### Physics Informed Neural Networks
This project will be applying physics informed machine learning techniques via Physics Informed Neural Networks (PINNs). PINNs, first introduced in [insert reference], operate much like traditional neural networks. Crucially, what makes a neural network a PINN is the network's loss function.
