import torch
import torch.nn.functional as F
from math import exp

def image_dim(img):
    return img.ndimension() - 2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, n_dim, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()

    _3D_window = torch.stack([ _2D_window * x for x in _1D_window],dim=2).float().unsqueeze(0).unsqueeze(0)
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)

    if n_dim == 3:
        return _3D_window.expand(channel, 1, window_size, window_size,window_size).contiguous()
    else:
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0

    n_dim = image_dim(img1)

    if n_dim == 2:
        (_, channel, height, width) = img1.size()
        convFunction = F.conv2d
    if n_dim == 3:
        convFunction = F.conv3d
        (_, channel, height, width, depth) = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, n_dim, channel=channel).to(img1.device)

    mu1 = convFunction(img1, window, padding=padd, groups=channel)
    mu2 = convFunction(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convFunction(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = convFunction(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = convFunction(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []

    n_dim = image_dim(img1)

    pool_size = [2] * n_dim
    if n_dim == 2:
        pool_function = F.avg_pool2d
    if n_dim == 3:
        pool_function = F.avg_pool3d

    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = pool_function(img1, pool_size)
        img2 = pool_function(img2, pool_size)

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        #Initialize window on first call
        if self.window is None:
            create_window(self.window_size, image_dim(img1)).to(img1.device).type(img1.dtype)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, image_dim(img1),channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
