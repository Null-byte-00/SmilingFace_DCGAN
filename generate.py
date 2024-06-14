from torchvision import transforms
from model import GAN
import torch


def generate():
    model = GAN()
    model.generator.load('models\\39_20240609015758\\generator.pth')
    out = model.generate().detach().cpu().squeeze(0)
    to_pil = transforms.ToPILImage()
    img = to_pil(out)
    img.show()


if __name__ == '__main__':
    generate()