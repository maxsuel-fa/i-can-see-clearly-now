import torch
import torch.nn as nn
import options.options as opt
from models.icscn import ICSCN
from torch.utils.data import DataLoader
from data.dataset import Dataset
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    options = opt.TrainingOptions()
    options = options.get_options()
    model = ICSCN(options, 'cpu', False)
    gen_path = options.save_dir + 'gen-test-2.pth'
    model.generator.load_state_dict(torch.load(gen_path))
    model.generator.requires_grad_(False)
    dataset = Dataset(options.clear_dir, options.rainy_dir, 'cpu', 10)
    dataloader = DataLoader(dataset, batch_size=1)
    transform = ToPILImage()
    for i, (_, rainy) in enumerate(dataloader):
        image = model.generator.forward(rainy)
        im_path = options.save_dir + 'img_cg' + str(i) + '.png' 
        save_image(image[0], im_path)
        
        #i
        #image.save(im_path, 'png')


    
    
