import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

class Transform:

    def __init__(self):
        self.train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

def real_loss(D_out, smooth = False, train_on_gpu = False):
    
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)*(1 - 0.1*smooth)
    
    if train_on_gpu:
        labels = labels.cuda()
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out, train_on_gpu):

    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def scale(x, feature_range=(-1, 1)):
    
    a, b = feature_range
    x = (b - a)*x + a
    return x
