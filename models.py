import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
     
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim = 32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # first layer, no batch_norm
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*2, 4)
        self.conv5 = conv(conv_dim*2, conv_dim*2, 4)
        self.fc1 = nn.Linear(conv_dim*2*8*8, 1024)
        self.fc2 = nn.Linear(1024, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = F.leaky_relu(self.conv5(out), 0.2)
        
        out = out.view(-1, self.conv_dim*2*8*8)
        out = F.leaky_relu(self.fc1(out), 0.2)
        out = F.leaky_relu(self.fc2(out), 0.2)
        out = self.fc3(out)        

        return out
    

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim = 32):
        super(Generator, self).__init__()
        
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        self.t_conv1 = deconv(conv_dim*4, conv_dim*4, 4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*4, 4)
        self.t_conv3 = deconv(conv_dim*4, conv_dim*4, 4)
        self.t_conv4 = deconv(conv_dim*4, conv_dim*4, 4)
        self.t_conv5 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv6 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv7 = deconv(conv_dim, 3, 4, batch_norm=False)
        

    def forward(self, x):
        
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)
        
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = F.relu(self.t_conv4(out))
        out = F.relu(self.t_conv5(out))
        out = F.relu(self.t_conv6(out))
        out = self.t_conv7(out)
        out = torch.tanh(out)
        
        return out