import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SDIM():
    def __init__(self, ):
        # Initialise SSIM object
        self.ssim_fn = StructuralSimilarityIndexMeasure(return_full_image=True)

    def forward(self, img, x_hat):
        # Add batch dimension if input is single image
        if img.dim() == 3:
            img = img.unsqueeze(0)
            x_hat = x_hat.unsqueeze(0)

        # Validate input dimensions
        assert img.dim() == 4, f"Expected 4D input (B,C,H,W), got {img.dim()}D"
        assert x_hat.shape == img.shape, f"Shape mismatch: {x_hat.shape} vs {img.shape}"

        # Get SSIM and similarity map for batch
        ssim_scores, ssim_maps = self.ssim_fn(img, x_hat)

        # Convert SSIM to SDIM scores - handle both scalar and vector cases
        if ssim_scores.dim() == 0:  # scalar case
            sdim_scores = [float(1 - ssim_scores.item()) / 2]
        else:  # vector case
            sdim_scores = [float(1 - score.item()) / 2 for score in ssim_scores]

        # Convert SSIM maps to anomaly heatmaps
        sdim_maps = torch.clamp(1 - ssim_maps.mean(dim=1), min=0, max=1) ** 2

        # If input is a single image, return single score and map
        if img.size(0) == 1:
            return sdim_scores[0], sdim_maps[0]
        return sdim_scores, sdim_maps
