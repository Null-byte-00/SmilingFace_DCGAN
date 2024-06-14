import torch
from torchvision import transforms
import matplotlib.pyplot as plt

dataset = torch.load("dataset/images.pth")

to_pil = transforms.ToPILImage()
fig = plt.figure(figsize=(11, 4))

columns = 5
rows = 2

for i in range(1,rows*columns + 1):
    fig.add_subplot(rows, columns, i)
    image, _ = dataset[i]
    plt.imshow(to_pil(image))

plt.show()