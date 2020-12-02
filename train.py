import argparse 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from utils import Transform, real_loss, fake_loss, scale
from models import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 32, help = 'Batch Size')
parser.add_argument('--epochs', type=int, help = 'Number of epochs')
parser.add_argument('--save_every', type = int, default = 5, help = 'Frequency of saving checkpoints')
parser.add_argument('--print_every', type = int, default = 5, help = 'Frequency of printing progress')
parser.add_argument('--resume_epoch', type = int, default = 0, help = 'Use when resuming training')
parser.add_argument('--cuda', type = bool, default = False, help = 'train on GPU')
parser.add_argument('--z_dim', type = int, default = 128, help = 'lateral dimension')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'learning rate')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'learning rate')
parser.add_argument('--beta2', type = float, default = 0.999, help = 'learning rate')
opt = parser.parse_args()

T = Transform()
data_dir = "../images"
dataset = datasets.ImageFolder(data_dir, transform = T.train_data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True)



gen = Generator(opt.z_dim)
disc = Discriminator()

if opt.cuda:
    gen = gen.cuda()
    disc = disc.cuda()

if opt.resume_epoch > 0:
    gen.load_state_dict(torch.load("checkpoints/gen_"+str(opt.resume_epoch)))
    disc.load_state_dict(torch.load("checkpoints/disc_"+str(opt.resume_epoch)))

d_optimizer = optim.Adam(disc.parameters(), opt.lr, [opt.beta1, opt.beta2])
g_optimizer = optim.Adam(gen.parameters(), opt.lr, [opt.beta1, opt.beta2])

for e in range(opt.resume_epoch, opt.epochs):

    for batch_i, (real_images, _) in enumerate(data_loader):

        batch_size = real_images.shape[0]
        real_images = scale(real_images)

        d_optimizer.zero_grad()

        z = np.random.uniform(-1, 1, size=(batch_size, opt.z_dim))
        z = torch.from_numpy(z).float()
        
        if opt.cuda:
            real_images = real_images.cuda()
            z = z.cuda()

        d_real = disc(real_images)
        d_real_loss = real_loss(d_real, True, opt.cuda)

        fake_images = gen(z)
        d_fake = disc(fake_images)
        d_fake_loss = fake_loss(d_fake, opt.cuda)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        z = np.random.uniform(-1, 1, size=(batch_size, opt.z_dim))
        z = torch.from_numpy(z).float()

        if opt.cuda:
            z = z.cuda()

        fake_images = gen(z)
        d_fake = disc(fake_images)
        g_loss = real_loss(d_fake, opt.cuda)


        g_optimizer.step()

        if (e - opt.resume_epoch) % opt.print_every == 0:
            print("Epoch:", e, "Batch: ", batch_i, " Generator Loss: ", g_loss, " Discriminator Loss: ", d_loss)









