from torchvision import datasets, transforms
import torch


def create_dataset(data_dir, dataset_name='dataset/images.pth'):
    transform = transforms.Compose([
        transforms.Resize((61, 61)),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    torch.save(dataset, dataset_name)


if __name__ == '__main__':
    create_dataset('dataset')