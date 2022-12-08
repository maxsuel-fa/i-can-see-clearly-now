import torch
import options.options as opt
from models.icscn import ICSCN
from torch.utils.data import DataLoader
from data.dataset import Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm


if __name__ == '__main__':
    options = opt.TrainingOptions()
    options = options.get_options()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ICSCN(options, device, is_train=True)
    dataset = Dataset(options.clear_dir, options.rainy_dir, device, 2000)
    dataloader = DataLoader(dataset, 
                            batch_size=options.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    # Silly test
    gen_path = options.save_dir + 'gen-test-2.pth'
    disc_path = options.save_dir + 'disc-test-2.pth'
    epochs = []
    losses = []
    dlosses = []
    loss = None
    im = None
    for epoch in tqdm(range(options.n_epochs), desc='epochs'):
        for i, (clear, rainy) in tqdm(enumerate(dataloader), desc='train'):
            model.forward(clear.to(device), rainy.to(device))
            model.optimize(options)
            #model.optimize(options)
            im = model.get_derained_image()
            #if(epoch % 100 == 0):
                #im = torch.squeeze(im, dim=0)
                #im = im.detach()
                #im = im.to('cpu')
                #im = im.permute(1, 2, 0)
                #print(im)
                #plt.imshow(im)
                #plt.pause(1e-9)
            #losses += [model.gen_loss.to('cpu').detach()]
            #epochs += [epoch]
            #plt.imshow(clear[0].permute(1, 2, 0))
            #print('pred_clear: \n', model.prceled_clear)
            #print('pred_rainy: \n', model.pred_rainy)
            #plt.plot(epochs, losses, 'r')
            #plt.plot(epochs, dlosses, 'b')
            #plt.pause(1e-9)
    torch.save(model.generator.state_dict(), gen_path)
    torch.save(model.discriminator.state_dict(), disc_path)
    

