import torch
import options.options as opt
from models.icscn import ICSCN
from torch.utils.data import DataLoader
from data.dataset import Dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    options = opt.TrainingOptions()
    options = options.get_options()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ICSCN(options, device, is_train=True)
    print(model.generator.state_dict())
    dataset = Dataset(options.clear_dir, options.rainy_dir, device)
    dataloader = DataLoader(dataset, batch_size=options.batch_size,
                     shuffle=True)
    for epoch in range(options.n_epochs):
        for i, (clear, rainy) in enumerate(dataloader):
            model.forward(clear[0], rainy[0])
            model.optimize(options)

    derained_image = model.get_derained_image()
    derained_image = torch.squeeze(derained_image, dim=0)
    derained_image = derained_image.permute(1, 2, 0)
    plt.imshow(derained_image)
    plt.show()
