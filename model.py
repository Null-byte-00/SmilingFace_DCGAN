import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels,4, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, out_channels=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.input_size = 4 * 286 * 4 

        self.linear = nn.Linear(latent_dim, self.input_size)

        self.model = nn.Sequential(
            Block(self.input_size, self.input_size // 2),
            Block(self.input_size // 2, self.input_size // 4),
            Block(self.input_size // 4, self.input_size // 8),
            Block(self.input_size // 8, self.input_size // 16),
            #Block(self.input_size // 16, self.input_size // 32),
            #Block(self.input_size // 32, self.input_size // 64),
            nn.ConvTranspose2d(self.input_size // 16, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, self.input_size, 1, 1)
        return self.model(x)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Discrimintor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=14, out_channels=28, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(4732, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.stack(x)


class GAN(nn.Module):
    def __init__(self, lr=0.000076,latent_dim = 100,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = Generator(latent_dim)
        self.discriminator = Discrimintor()
        self.latent_dim = latent_dim

        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.loss = nn.BCELoss()
    
    def forward(self, z):
        x = self.generator(z)
        return self.discriminator(x)

    def train_G(self):
        self.generator.zero_grad()
        z = torch.randn(1, self.latent_dim).to(self.generator.linear.weight.device)
        g_output = self.generator(z)
        d_fake_output = self.discriminator(g_output)
        loss = self.loss(d_fake_output, torch.ones_like(d_fake_output))
        loss.backward()
        self.optim_g.step()
        return loss

    def train_D(self, x):
        self.discriminator.zero_grad()

        z = torch.randn(1, self.latent_dim).to(self.generator.linear.weight.device) # noise
        g_output = self.generator(z)
        d_fake_output = self.discriminator(g_output)
        loss_fake = self.loss(d_fake_output, torch.zeros_like(d_fake_output))

        d_real_output = self.discriminator(x)
        loss_real = self.loss(d_real_output, torch.ones_like(d_real_output))

        loss = loss_fake + loss_real
        loss.backward()
        self.optim_d.step()

        return loss
    
    def train(self, x, verbose=False):
        loss1 = self.train_D(x)
        loss2 = self.train_G()
        if verbose:
            print(f'Loss D: {loss1.item()} Loss G: {loss2.item()}')
        return loss1, loss2
    
    def generate(self):
        z = torch.randn(1, self.latent_dim).to(self.generator.linear.weight.device)
        return self.generator(z)

    def save(self, path):
        torch.save(self.generator.state_dict(), f'{path}/generator.pth')
        torch.save(self.discriminator.state_dict(), f'{path}/discriminator.pth')
    
    def load(self, path):
        self.generator.load(f'{path}/generator.pth')
        self.discriminator.load(f'{path}/discriminator.pth')
    
    def plot(self):
        to_pil = transforms.ToPILImage()
        fig = plt.figure(figsize=(11, 4))

        columns = 5
        rows = 1

        for i in range(1,rows*columns + 1):
            fig.add_subplot(rows, columns, i)
            image = self.generate().detach().cpu().squeeze(0)
            plt.imshow(to_pil(image), cmap='gray')
        plt.show()