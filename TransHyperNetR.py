import torch
import torch.nn as nn
import torch.nn.functional as F

def hyperNet(x_dim=3, y_dim=3):
    xx_range = torch.arange(-(x_dim - 1) / 2, (x_dim + 1) / 2, dtype=torch.float32)
    yy_range = torch.arange(-(y_dim - 1) / 2, (y_dim + 1) / 2, dtype=torch.float32)

    xx_range = xx_range.unsqueeze(-1).repeat(1, y_dim)
    yy_range = yy_range.unsqueeze(0).repeat(x_dim, 1)

    xx_range = xx_range.unsqueeze(-1)
    yy_range = yy_range.unsqueeze(-1)

    pos = torch.cat([xx_range, yy_range], dim=-1)
    pos = pos.unsqueeze(0)

    return pos

def p_conv(ip, kernal):
  # Reshape the kernel to remove the channel dimension (originally 1)
  ip = ip.view(ip.shape[0], -1, ip.shape[2], ip.shape[3])  # Reshape to 4D
  # Reshape the kernel to have the same number of input channels as ip
  kernal = kernal.view(-1, ip.shape[1], kernal.shape[2], kernal.shape[3])
  out = F.conv2d(ip, kernal, padding='same')
  return out



class PKernel(nn.Module):
    def __init__(self, x_dim, y_dim, ch_in, ch_out):
        super().__init__()
        self.num_c = ch_in * ch_out
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(16, 4, kernel_size=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(4, self.num_c, kernel_size=1, padding='same', bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ip):
        pos = self.leaky_relu(self.conv1(ip))
        pos = self.leaky_relu(self.conv2(pos))
        pos = self.leaky_relu(self.conv3(pos))
        pos = self.conv4(pos)
        pos = pos.view(pos.shape[0], pos.shape[1], pos.shape[2], -1)
        return pos


class HyperBlock(nn.Module):
    def __init__(self, x_dim, y_dim, ch_in, ch_out, act='relu', bn=False, do=0, multi=True, res=False):
        super().__init__()
        self.input_channel = ch_in
        self.kernal1 = hyperNet(x_dim, y_dim)
        self.p_kernal1 = PKernel(x_dim, y_dim, ch_in, ch_out)
        self.bn = bn
        self.do = do
        self.multi = multi
        self.res = res
        self.activation = nn.ReLU() if act == 'relu' else nn.Identity()

        if multi:
            self.kernal2 = hyperNet(x_dim, y_dim)
            self.p_kernal2 = PKernel(x_dim, y_dim, ch_out, ch_out)

        if bn:
            self.bn1 = nn.BatchNorm2d(ch_out)
            if multi:
                self.bn2 = nn.BatchNorm2d(ch_out)

        if do:
            self.dropout = nn.Dropout(do)

    def forward(self, ip):
        kernal1 = self.p_kernal1(self.kernal1)
        n = p_conv(ip, kernal1)
        if self.bn:
            n = self.bn1(n)
        n = self.activation(n)
        if self.do:
            n = self.dropout(n)

        if self.multi:
            kernal2 = self.p_kernal2(self.kernal2)
            n = p_conv(n, kernal2)
            if self.bn:
                n = self.bn2(n)
            n = self.activation(n)

        return torch.cat([ip, n], dim=1) if self.res else n
