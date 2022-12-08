import torch
import torch.utils.data as data
from torchvision.transforms import Resize, ToTensor, Compose, RandomCrop
from torch import Tensor
import os
from PIL import Image

def get_paths(dir: str, max_dataset_size=10000) -> list:
    paths = []
    for dir_name, _, file_names in os.walk(dir):
        for file_name in sorted(file_names):
            paths += [os.path.join(dir_name, file_name)] 
    dataset_len = len(paths)
    return paths[:min(dataset_len, max_dataset_size)]

class Dataset(data.Dataset):
    def __init__(self, clear_dir: str, rainy_dir: str, 
                 device: str, max_dataset_size: int) -> None:
        super(Dataset, self).__init__()
        self.clear_paths = get_paths(clear_dir, max_dataset_size)
        self.rainy_paths = get_paths(rainy_dir, max_dataset_size)
        transforms = [Resize((688, 768)), ToTensor()]
        self.transform = Compose(transforms)
        self.device = device

    def __len__(self) -> int:
        return len(self.clear_paths)

    def __getitem__(self, index) -> tuple:
        clear_image = Image.open(self.clear_paths[index])
        clear_image = self.transform(clear_image.convert('RGB'))

        rainy_image = Image.open(self.rainy_paths[index])
        rainy_image = self.transform(rainy_image.convert('RGB'))

        return clear_image, rainy_image
