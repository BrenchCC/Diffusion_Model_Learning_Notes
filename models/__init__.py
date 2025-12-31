import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the time step t into a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: tensor of shape (batch_size,)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
        return embeddings

class Block(nn.Module):
    """
    ResNet-style block with Time Embedding injection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, groups = 8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: Feature map (B, C, H, W)
            t_emb: Time embedding (B, dim)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    """
    Simple U-Net architecture for DDPM demo.
    """
    def __init__(
        self, 
        image_channels = 3, 
        down_channels = (64, 128, 256), 
        up_channels = (256, 128, 64), 
        out_dim = 3, 
        time_emb_dim = 128
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding = 1)
        
        self.downs = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.downs.append(Block(down_channels[i], down_channels[i+1], time_emb_dim))
            
        self.ups = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(up_channels[i], up_channels[i+1], 2, stride = 2),
                Block(up_channels[i] + up_channels[i+1], up_channels[i+1], time_emb_dim)
            ]))
            
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        """
        Args:
            x: Input tensor
            timestep: Time step tensor
        """
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
            x = F.max_pool2d(x, 2)
            
        for up_trans, up_block in self.ups:
            residual = residuals.pop()
            x = up_trans(x)
            if x.shape != residual.shape:
                x = F.interpolate(x, size = residual.shape[2:])
            x = torch.cat((x, residual), dim = 1)
            x = up_block(x, t)
            
        return self.output(x)