from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os


def load_transformed_dataset(img_size=256, batch_size=128) -> DataLoader:
    # Load dataset and perform data transformations
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # TODO: 你可以更改这两个地方的路径，以实现对其他数据集的加载
    # 当然，你也可以添加更多的参数，以支持不同数据集之间的修改
    train = torchvision.datasets.ImageFolder(root="./datasets-2/train", transform=data_transform)

    test = torchvision.datasets.ImageFolder(root="./datasets-2/test", transform=data_transform)

    dataset = torch.utils.data.ConcatDataset([train, test])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def show_tensor_image(image, i=None, mode="generation"):
    save_dir = mode
    os.makedirs(save_dir, exist_ok=True)

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.clamp(0, 1)),
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
    ])

    img = reverse_transforms(image)

    if i is None:
        filename = "result.png"
    else:
        filename = f"{int(i):04d}.png"

    save_path = os.path.join(save_dir, filename)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path