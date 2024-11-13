import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from model import Generator
from utils import save_models


class WDiscriminator(nn.Module):
    '''Return an output that is not a probability.
    In WGAN we try to compare distances thus real are annotated as 1 and fake as -1.'''
    def __init__(self, d_input_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        # no sigmoid because no probability
        return x

class WGAN():
    def __init__(self, lr=0.00005, n_critic=5, clip_value=0.01, batch_size=2048):
        self.mnist_dim = 784
        self.lr = lr
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = torch.nn.DataParallel(Generator(g_output_dim = self.mnist_dim)).to(self.device)
        self.D = torch.nn.DataParallel(WDiscriminator(self.mnist_dim)).to(self.device)
        self.G_optimizer = torch.optim.RMSprop(self.G.parameters(), lr = lr)
        self.D_optimizer = torch.optim.RMSprop(self.D.parameters(), lr = lr)
        self.criterion = nn.HingeEmbeddingLoss()

    def train(self, train_loader, n_epoch):

        iter = self.get_iter(train_loader)

        fake_losses = []
        real_losses = []
        G_losses = []

        # For basic WGAN
        one = torch.FloatTensor([1]).to(self.device)
        mone = one * -1
        mone = mone.to(self.device)

        for epoch in range(n_epoch):
            print('epoch: ', epoch)
            for _ in tqdm.tqdm(range(len(train_loader)//self.n_critic)):
                for p in self.D.parameters():
                    p.requires_grad = True

                for c_iter in range(self.n_critic):
                    for p in self.D.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                    self.D.zero_grad()

                    x = iter.__next__().to(self.device)
                    # Check for batch to have full batch_size
                    if (x.size()[0] != self.batch_size):
                        continue
                    x = x.view(-1, self.mnist_dim)

                    y_real = torch.ones(self.batch_size, 1)
                    y_real.requires_grad = True
                    # again, we are not interested in probabilities so we use -1 to represent fake
                    y_fake = torch.ones(self.batch_size, 1) * -1
                    y_fake.requires_grad = True
                    y_real = y_real.to(self.device)
                    y_fake = y_fake.to(self.device)

                    D_real = self.D(x)
                    # For basic WGAN
                    D_real = D_real.mean(0).view(1)
                    D_real.backward(one)
                    # D_real = self.criterion(y_real, D_real)
                    # D_real.backward()


                    z = torch.randn(x.shape[0], 100).to(self.device)
                    G_z = self.G(z)
                    D_fake = self.D(G_z)
                    # For basic WGAN
                    D_fake = D_fake.mean(0).view(1)
                    D_fake.backward(mone)
                    # D_fake = self.criterion(y_fake, D_fake)
                    # D_fake.backward()
                    fake_losses += [D_fake.item()]
                    real_losses += [D_real.item()]
                    D_loss = D_fake - D_real
                    self.D_optimizer.step()

                for p in self.D.parameters():
                    # to avoid computation
                    p.requires_grad = False

                self.G.zero_grad()
                z = torch.randn(self.batch_size, 100).to(self.device)
                G_z = self.G(z)
                D_fake = self.D(G_z)
                G_loss = D_fake.mean(0).view(1)
                G_loss.backward(one)
                # G_loss = self.criterion(D_fake, y_real)
                # G_loss.backward()
                G_losses += [G_loss.item()]
                self.G_optimizer.step()

            if epoch % 10 == 0:
                self.save_models('checkpoints')
        return G_losses, fake_losses, real_losses


    def get_iter(self, data_loader):
      "allow for infinite iterations"
      while True:
        for i, (x, _) in enumerate(data_loader):
            yield x

    def generate(self, n):
        z = torch.randn(n, 100).to(self.device)
        return self.G(z)

    def save_models(self, folder):
      os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
      torch.save(self.G.state_dict(), os.path.join(folder, 'WG.pth'))
      torch.save(self.D.state_dict(), os.path.join(folder, 'WD.pth'))

    def to_np(self, x):
        return x.data.cpu().numpy()