import torch 
import torchvision
import os
import argparse


from model import Generator
from utils import load_model

import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.nn.utils import spectral_norm

import torchvision.utils as vutils
import shutil

class Generator(nn.Module):
    def __init__(self, g_output_dim=784, dim_latent=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class BaseDiscriminator(nn.Module):
    def __init__(self, d_input_dim=784):
        super(BaseDiscriminator, self).__init__()
        self.h_function = nn.Sequential(
            #nn.Linear(d_input_dim, 512),
            spectral_norm(nn.Linear(d_input_dim, 1024)),
            nn.LeakyReLU(0.2),
            #nn.Linear(1024, 512),
            #nn.LeakyReLU(0.2),
            #nn.Linear(512, 256),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            #nn.Linear(256, 1)
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
        )

        self.fc_w = nn.Parameter(torch.randn(1, 256))

    def forward(self, x, flg_train: bool):        
        h_feature = self.h_function(x)
        weights = self.fc_w
        out = (h_feature * weights).sum(dim=1)
        return out

class SanDiscriminator(BaseDiscriminator):
    def __init__(self, d_input_dim=784):
        super(SanDiscriminator, self).__init__(d_input_dim)

    def forward(self, x, flg_train: bool):
        h_feature = self.h_function(x)        
        weights = self.fc_w
        direction = F.normalize(weights, dim=1)  # Normalize the last layer
        scale = torch.norm(weights, dim=1).unsqueeze(1)
        h_feature = h_feature * scale  # Keep the scale
        if flg_train:  # For discriminator training
            out_fun = (h_feature * direction.detach()).sum(dim=1)
            out_dir = (h_feature.detach() * direction).sum(dim=1)
            out = dict(fun=out_fun, dir=out_dir)
        else:  # For generator training or inference
            out = (h_feature * direction).sum(dim=1)
        return out

def generate_samples_with_DRS(G, D, num_samples, batch_size, tau):
    # Thanks Gangineers ;)
    G.eval()
    D.eval()
    samples = []
    total_generated = 0
    total_attempted = 0

    while total_generated < num_samples:
        z = torch.randn(batch_size, 100).cuda()
        with torch.no_grad():
            x_fake = G(z)
            D_output = D(x_fake, flg_train=False).squeeze()
            acceptance_probs = torch.sigmoid(D_output - tau)
            accept = torch.bernoulli(acceptance_probs).bool()
            accepted_samples = x_fake[accept]
            samples.append(accepted_samples.cpu())
            total_generated += accepted_samples.size(0)
            total_attempted += batch_size
    acceptance_rate = total_generated / total_attempted
    print(f'Acceptance Rate: {acceptance_rate:.4f}')

    G.train()
    D.train()

    samples = torch.cat(samples, dim=0)
    samples = samples[:num_samples]

    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    G = Generator()
    D = SanDiscriminator()

    device = torch.device("cuda")
    G = G.to(device)
    D = D.to(device)

    # Load the state_dict
    G.load_state_dict(torch.load('checkpoints/generator_san.pth'))
    D.load_state_dict(torch.load('checkpoints/discriminator_san.pth'))
    G.eval()
    D.eval()

    tau = 0.9
    num_samples = 10000
    batch_size = 256

    print(f'Generating {num_samples} samples with DRS (tau={tau})...')
    samples = generate_samples_with_DRS(G, D, num_samples, batch_size, tau)

    folder_name = 'samples'

    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    for idx in range(samples.size(0)):
        vutils.save_image(samples[idx].view(1, 28, 28),
                        f'samples_drs/{idx}.png',
                        normalize=True)

"""
    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1
"""