from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.datasets.vision as vision
import torchvision.io as io
import torch.optim as optim

from typing import Any, TypeVar, Callable, Dict, List, Optional, Tuple


Tensor = TypeVar("torch.tensor")


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 in_spatial_H = 256,
                 in_spatial_W = 256,
                 num_hidden_layers = 5,
                 hidden_dims: List = [32, 64, 128, 256, 512],
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        assert len(hidden_dims) == num_hidden_layers

        self.H = in_spatial_H // (2**num_hidden_layers)
        self.W = in_spatial_W // (2**num_hidden_layers)
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.H * self.W, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.H * self.W, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.H * self.W)

        hidden_dims = self.hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size= 3, padding= 1),
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], self.hidden_dims[-1], self.H, self.W)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class DAVIS(vision.VisionDataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 10,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_frames = num_frames
        self.train = train  # training set or test set
        self.root = root

        self.video_names = []
        if train:
            with open(os.path.join(root, 'ImageSets/2017/train.txt'), 'r') as f:
                for line in f:
                    self.video_names.append(line.strip('\n'))

        else:
            with open(os.path.join(root, 'ImageSets/2017/val.txt'), 'r') as f:
                for line in f:
                    self.video_names.append(line.strip('\n'))

    def __getitem__(self, index: int) -> Any:
        video_name = self.video_names[index]
        video_frames = list((Path(self.root)/'JPEGImages/480p/{}'.format(video_name)).glob('*.jpg'))
        video_frames.sort()

        start_frame = random.randrange(len(video_frames) - self.num_frames)
        video_frames = video_frames[start_frame: start_frame + self.num_frames]    

        video_frames_data = self._load_data(video_frames)
        video_frames_target = 0 # we don't need target

        # random crop
        random_crop = transforms.RandomCrop((480, 840), pad_if_needed=True)
        resize = transforms.Resize((256, 512))
        video_frames_data = resize(random_crop(video_frames_data))
        video_frames_data = video_frames_data / 255 # maximum value for all images in DAVIS

        return video_frames_data, video_frames_target

    def __len__(self):
        return len(self.video_names)

    def _load_data(self, video_frames):
        video_frames_data = []
        for video_frame in video_frames:
            frame_data = io.read_image(str(video_frame), mode=io.ImageReadMode.RGB)
            video_frames_data.append(frame_data)
        return torch.stack(video_frames_data, dim=0).float()


## Dataset Parameters
davis_root = '/playpen-raid2/qinliu/data/DAVIS'
mb_size = 1 # each batch use only one video 
num_frames = 10 # each video uses 10 consecutive frames for training

## Training Parameters
in_channels = 3
latent_dim = 128
c = 0
lr = 1e-4
device = torch.device("cuda:3")


# Build dataset
trainset = DAVIS(root=davis_root, num_frames=num_frames, train=True)
valset = DAVIS(root=davis_root, num_frames=1, train=False)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=mb_size, 
                                           shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(valset, batch_size=mb_size, 
                                          shuffle=False, num_workers=1)


model = VanillaVAE(in_channels, latent_dim, num_frames=num_frames, in_spatial_H=256, in_spatial_W=512, \
    num_hidden_layers=6, hidden_dims=[8, 16, 32, 64, 128, 256])
model.to(device)

solver = optim.Adam(model.parameters(), lr=lr)

for it in range(100000):
    X, _ = next(iter(train_loader))
    X = torch.squeeze(X, dim=0)
    X = X.to(device)

    # Forward
    z_mu, z_var = model.encode(X)    

    z = model.reparameterize(z_mu, z_var)
    X_sample = model.decode(z)

    # Loss
    recon_loss = F.l1_loss(X_sample, X, reduction='sum') / num_frames
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    solver.zero_grad()

    # Print and plot every now and then
    if it % 200 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.item()))

        samples = model.decode(z)
        samples = torch.sigmoid(samples)
        samples = samples.cpu().detach().numpy()[:9]

        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            sample = np.moveaxis(sample, 0, -1)
            plt.imshow(sample)

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)