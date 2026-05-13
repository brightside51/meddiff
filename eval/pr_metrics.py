import torch
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from tqdm.auto import tqdm

class InceptionV3_FeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model: torch.nn.Module = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(2048)],
            feature_extractor_weights_path=None,
        ).to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = (imgs * 255).byte()
        feats = self.model(imgs)
        feats = feats.double()
        return feats

class RadImageNet_FeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model: torch.nn.Module = RadImageNetPerceptualSimilarity().model.to(device)
        self.device = device

    def _subtract_mean(self, x: torch.Tensor) -> torch.Tensor:
        mean = [0.406, 0.456, 0.485]
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        return x

    def _normalize_tensor(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)
    
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = self._subtract_mean(imgs)
        feats = self.model.forward(imgs)
        feats = self._normalize_tensor(feats).mean(dim=(2,3)).double()
        return feats


class PrecisionRecall:
    """
    Implementation of the pseudocode in https://arxiv.org/pdf/1904.06991
    """
    def __init__(self, device, k: int, feature_extractor: str):
        self.feature_extractor = InceptionV3_FeatureExtractor(device) if feature_extractor=="IV3" else RadImageNet_FeatureExtractor(device)
        self.k = k

    def _manifold_estimate(self, Phi_a: torch.Tensor, Phi_b: torch.Tensor, k: int):
        # pairwise distances between all elements in Phi_a
        dists = torch.cdist(Phi_a, Phi_a, p=2) # shape (B_a, B_a)

        r_phi, _ = torch.topk(dists, k+1, dim=1, largest=False)
        r_phi = r_phi[:, -1]

        # Compute pairwise distances from Phi_b to Phi_a
        dists_ba = torch.cdist(Phi_b, Phi_a, p=2) # shape (B_b, B_a)

        # Check if any distance to Phi_a is <= corresponding r_phi
        within_manifold = (dists_ba <= r_phi.unsqueeze(0))  # shape (B_b, B_a)

        # a Phi_b point is inside the manifold if any of the distances are below r_phi
        is_inside = within_manifold.any(dim=1)  # shape: (B_b,)

        return is_inside.float().mean()
    
    def compute(self, real_samples: torch.Tensor, fake_samples: torch.Tensor):
        phi_real = self.feature_extractor.forward(real_samples)
        phi_fake = self.feature_extractor.forward(fake_samples)
        
        precision_score = self._manifold_estimate(phi_real, phi_fake, self.k)
        recall_score = self._manifold_estimate(phi_fake, phi_real, self.k)

        return precision_score, recall_score
        

@torch.no_grad()
def mmd_metric(original_collection: torch.Tensor, sample_colletion: torch.Tensor, feature_extractor: str = "IV3", device: torch.device = 'cpu'):
    feature_extractor = InceptionV3_FeatureExtractor(device) if feature_extractor=="IV3" else RadImageNet_FeatureExtractor(device)

    # Gaussian RBF kernel (vectorized)
    def gaussian_rbf_kernel(x, y, sigma=10):
        diff = x[:, None, :] - y[None, :, :]  # Shape: (Nx, Ny, D)
        dist_sq = torch.sum(diff**2, dim=-1)  # Squared L2 distance: Shape (Nx, Ny)
        return torch.exp(-dist_sq / (2 * sigma**2))  # Gaussian kernel

    # Extract features
    X = feature_extractor.forward(original_collection)
    Y = feature_extractor.forward(sample_colletion)

    # Flatten features
    X = X.view(X.shape[0], -1)  # Shape: (Nx, D)
    Y = Y.view(Y.shape[0], -1)  # Shape: (Ny, D)

    # Compute kernel matrices
    K_XX = gaussian_rbf_kernel(X, X)  # Shape: (Nx, Nx)
    K_YY = gaussian_rbf_kernel(Y, Y)  # Shape: (Ny, Ny)
    K_XY = gaussian_rbf_kernel(X, Y)  # Shape: (Nx, Ny)

    # Compute the MMD
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    # Ensure non-negativity for numerical stability
    return mmd.item()

@torch.no_grad()
def precision_recall_metric(original_collection: torch.Tensor, sample_colletion: torch.Tensor, feature_extractor: str = "IV3", device: torch.device = 'cpu'):
    precision_recall = PrecisionRecall(device, k=3, feature_extractor=feature_extractor)
    precision, recall = precision_recall.compute(original_collection, sample_colletion)
    return precision.item(), recall.item()

@torch.no_grad()
def ms_ssim_metric(original_colletion: torch.Tensor, sample_colletion: torch.Tensor, device: torch.device = 'cpu'):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    ms_ssim.update(sample_colletion, original_colletion)
    return ms_ssim.compute().item()

@torch.no_grad()
def ms_ssim_variety_metric(sample_colletion: torch.Tensor, device: torch.device = 'cpu'):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    for i in tqdm(range(0, sample_colletion.shape[0]), desc="MS-SSIM"):
        for j in range(i, sample_colletion.shape[0]):
            ms_ssim.update(sample_colletion[i:i+1, ...], sample_colletion[j:j+1, ...])
    return ms_ssim.compute().item()

@torch.no_grad()
def psnr_metric(original_collection: torch.Tensor, sample_colletion: torch.Tensor, device: torch.device = 'cpu'):
    psnr = PeakSignalNoiseRatio().to(device)
    return psnr(sample_colletion, original_collection).item()