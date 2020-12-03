import argparse
import matplotlib.pyplot as plt 
from models import Generator
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type = int, help = 'num of epoch for checkpoints to visualize results')
opt = parser.parse_args()

z_size = 128
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16, 16), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach()
        img = img.cpu()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((256, 256, 3)))

gen = Generator(z_size)

gen.load_state_dict(torch.load("checkpoints/gen_"+str(opt.num_epoch) + ".pt"))

