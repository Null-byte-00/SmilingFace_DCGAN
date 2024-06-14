import torch
import os
from datetime import datetime
from model import GAN


def train(model, dataset='dataset/images.pth', epochs=1, verbose=True, device='cuda'):
    dataset = torch.load(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model.to(device)

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for i, (x, _) in enumerate(dataloader):
            if verbose:
                print(f'Batch {i + 1}/{len(dataloader)} Epoch {epoch + 1}/{epochs}')
            x = x.to(device)
            loss_d, loss_g = model.train(x.clone().detach(), verbose=verbose)
            d_losses.append(loss_d)
            g_losses.append(loss_g)
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)

        if (epoch + 1) % 3 == 0:
            print(f"epoch {epoch + 1} Discriminator loss: {avg_d_loss} Generator loss: {avg_g_loss}")
            model.plot()
            path = f'models/{epoch}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
            os.makedirs(path, exist_ok=True)
            model.save(path)


if __name__ == '__main__':
    model = GAN()
    train(model, epochs=10)