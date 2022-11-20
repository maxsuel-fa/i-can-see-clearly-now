import torch
import torch.utils.data as data
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch import Tensor
import os

def get_paths(dir: str) -> list:
    paths = []
    for dir_name, _, file_names in os.walk(dir):
        for file_name in sorted(file_names):
            paths += [os.path.join(dir_name, file_name)] 
    return paths

class Dataset(data.Dataset):
    def __init__(self, clear_dir: str, rainy_dir: str, device: str) -> None:
        super(Dataset, self).__init__()
        self.clear_paths = get_paths(clear_dir)
        self.rainy_paths = get_paths(rainy_dir)
        self.transform = Resize((688, 768))
        self.device = device

    def __len__(self) -> int:
        return len(self.clear_paths)

    def __getitem__(self, index) -> tuple:
        clear_image = read_image(self.clear_paths[index])
        clear_image = torch.unsqueeze(clear_image, dim=0)
        clear_image = self.transform(clear_image)
        clear_image = clear_image.to(self.device, dtype=torch.float)

        rainy_image = read_image(self.rainy_paths[index])
        rainy_image = torch.unsqueeze(rainy_image, dim=0)
        rainy_image = self.transform(rainy_image)
        rainy_image = rainy_image.to(self.device, dtype=torch.float)

        return clear_image, rainy_image


