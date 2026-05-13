import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import scipy.ndimage as ndimage
#import filters
#import skimage.filters as filters

"""
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    #_3D_window = filters.gaussian_filter
    return window

def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = ndimage.gaussian_filter(img1, sd)
    mu2 = ndimage.gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = ndimage.gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = ndimage.gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = ndimage.gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = np.mean(v1 / v2)  # contrast sensitivity

#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return np.mean(ssim_map), cs


def ssim_3d(img1, img2, kernel, size_average=True):
    #Calculates the SSIM for 3D images.
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.conv3d(img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1))
    mu2 = F.conv3d(img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map

def ssim_3d(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
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
    (_, channel, height, width, width2) = img1.size()
    if window is None:
        real_size = min(window_size, height, width, width2)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim_3d(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim_exact(img1.data.cpu().numpy(), img2.data.cpu().numpy())
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool3d(img1, (2, 2, 2))
        img2 = F.avg_pool3d(img2, (2, 2, 2))

    mssim = np.asarray(mssim)
    mcs = np.asarray(mcs)

    mssim = torch.from_numpy(mssim)
    mcs = torch.from_numpy(mcs)
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
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
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(
                img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM_3d(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM_3d, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim_3d(img1, img2, window_size=self.window_size, size_average=self.size_average)

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
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size,
                       size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
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
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(
                img1.device).type(img1.dtype)
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
    
"""

def gaussian_kernel(size: int, sigma: float):
    """Creates a 3D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return kernel

def create_gaussian_window(window_size, sigma, channels):
    """
    Create a 3D Gaussian kernel.
    Args:
        window_size (int): Size of the Gaussian kernel (odd number).
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of input channels.
    Returns:
        torch.Tensor: Gaussian kernel for 3D convolution.
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    grid = torch.stack(torch.meshgrid(coords, coords, coords), dim=-1)
    kernel = torch.exp(-torch.sum(grid**2, dim=-1) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize kernel
    kernel = kernel.view(1, 1, window_size, window_size, window_size)
    return kernel.expand(channels, 1, -1, -1, -1)

def SSIM(img1, img2, window_size=3, sigma=1.5, data_range=1.0):
    """
    Compute 3D Structural Similarity Index (SSIM) between two volumes.
    Args:
        img1 (torch.Tensor): First input tensor of shape (B, C, D, H, W).
        img2 (torch.Tensor): Second input tensor of shape (B, C, D, H, W).
        window_size (int): Size of the Gaussian kernel (odd number).
        sigma (float): Standard deviation of the Gaussian kernel.
        data_range (float): Maximum possible value in the data (e.g., 1.0 if normalized).
    Returns:
        torch.Tensor: Mean SSIM value over the batch.
    """
    assert img1.shape == img2.shape, "Input volumes must have the same shape!"
    B, C, D, H, W = img1.shape
    device = img1.device

    # Create Gaussian kernel
    kernel = create_gaussian_window(window_size, sigma, C).to(device)

    # Compute means
    mu1 = F.conv3d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv3d(img2, kernel, padding=window_size // 2, groups=C)

    # Compute variances and covariances
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # Return mean SSIM over the batch
    return ssim_map.mean()

def _ssim_3d(img1, img2, kernel, size_average=True):
    """Calculates the SSIM for 3D images."""
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.conv3d(img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1))
    mu2 = F.conv3d(img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map

def MS_SSIM(img1, img2, levels=5, size_average=True):
    """Calculates the Multi-Scale SSIM for 3D images."""
    img_size = img1.shape[-1]  # Assuming shape is (B, C, D, H, W)
    pad = torch.zeros(1, img_size, img_size).to(img1.device)
    img1 = torch.concat([pad, img1[0, 0], pad]).unsqueeze(0).unsqueeze(0)
    img2 = torch.concat([pad, img2[0, 0], pad]).unsqueeze(0).unsqueeze(0)
    #print(f"Padding: {img1.shape}")
    kernel = gaussian_kernel(size=3, sigma=1.5).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(img1.device)
    
    mssim = []
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)

    for _ in range(levels):
        ssim = _ssim_3d(img1, img2, kernel, size_average)
        mssim.append(ssim)

        img1 = F.avg_pool3d(img1, kernel_size=2)
        img2 = F.avg_pool3d(img2, kernel_size=2)

    mssim = torch.stack(mssim)
    return (mssim * weights).sum() if size_average else mssim