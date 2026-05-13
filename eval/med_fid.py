import torch
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from monai.losses.perceptual import RadImageNetPerceptualSimilarity

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


class FID_RadImageNet:
    def __init__(self, device):
        self.feature_extractor = RadImageNet_FeatureExtractor(device)
        
        self.real_features_sum = torch.zeros((2048,)).double().to(device)
        self.real_features_cov_sum = torch.zeros((2048, 2048)).double().to(device)
        self.real_features_num_samples = 0.0

        self.fake_features_sum = torch.zeros((2048,)).double().to(device)
        self.fake_features_cov_sum = torch.zeros((2048, 2048)).double().to(device)
        self.fake_features_num_samples = 0.0
    
    def _compute_fid(self, mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor: 
        a = (mu1 - mu2).square().sum(dim=-1)
        b = sigma1.trace() + sigma2.trace()
        c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
        return a + b - 2 * c

    @torch.no_grad()
    def update(self, imgs: torch.Tensor, real: bool):
        feats = self.feature_extractor.forward(imgs)

        if real:
            self.real_features_sum += feats.sum(dim=0)
            self.real_features_cov_sum += feats.t().mm(feats)
            self.real_features_num_samples += feats.shape[0]
        else:
            self.fake_features_sum += feats.sum(dim=0)
            self.fake_features_cov_sum += feats.t().mm(feats)
            self.fake_features_num_samples += feats.shape[0]

    @torch.no_grad()
    def compute(self):
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        return self._compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)

@torch.no_grad()
def fid_metric(original_collection: torch.Tensor, sample_colletion: torch.Tensor, feature_extractor: str = "IV3", device: torch.device = 'cpu'):
    fid = FrechetInceptionDistance(normalize=True).to(device) if feature_extractor=="IV3" else FID_RadImageNet(device)
    original_collection = original_collection.repeat(1, 3, 1, 1)
    sample_colletion = sample_colletion.repeat(1, 3, 1, 1)
    for i in tqdm(range(original_collection.shape[0]), desc=f"FID_{feature_extractor}"):
        fid.update(original_collection[i:i+1, ...], real=True)
        fid.update(sample_colletion[i:i+1, ...], real=False)
    return fid.compute().item()