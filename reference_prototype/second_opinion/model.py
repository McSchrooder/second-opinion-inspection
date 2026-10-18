"""PatchCore-style anomaly detector (Roth et al., 2022, "Towards Total Recall in
Industrial Anomaly Detection"), written small and readable on purpose.

Idea: keep a memory bank of patch features from GOOD images only. A new image is
scored by how far each of its patches is from the nearest good patch. The patch
distances form the heatmap; the largest one is the image score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _backbone(name: str, weights_path: str | None) -> torch.nn.Module:
    ctor = getattr(torchvision.models, name)
    net = ctor(weights=None)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.pretrained_ok = True
    else:
        try:
            net = ctor(weights="DEFAULT")
            net.pretrained_ok = True
        except Exception:  # no internet: random weights, results only useful as a smoke test
            net.pretrained_ok = False
    return net.eval()


class PatchCore:
    def __init__(self, backbone: str = "wide_resnet50_2", weights_path: str | None = None,
                 image_size: int = 224, coreset_ratio: float = 0.1, proj_dim: int = 256,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.net = _backbone(backbone, weights_path)
        self.pretrained = bool(getattr(self.net, "pretrained_ok", False))
        self.image_size = image_size
        self.coreset_ratio = coreset_ratio
        self.proj_dim = proj_dim
        self.proj: torch.Tensor | None = None
        self.bank: torch.Tensor | None = None
        self.seed = seed

    # ---------- features ----------
    @torch.no_grad()
    def _features(self, images: np.ndarray) -> torch.Tensor:
        """images: N x H x W x 3 uint8 -> N x D x h x w patch features (layer2 + layer3)."""
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l1 = n.layer1(x)
        l2 = n.layer2(l1)
        l3 = n.layer3(l2)
        # local neighbourhood aggregation, as in PatchCore
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([l2, l3], dim=1)
        if self.proj is None:
            g = torch.Generator().manual_seed(self.seed)
            self.proj = torch.randn(feats.shape[1], self.proj_dim, generator=g) / self.proj_dim ** 0.5
        return torch.einsum("ndhw,dk->nkhw", feats, self.proj)

    def _batched(self, images: np.ndarray, bs: int = 8) -> torch.Tensor:
        return torch.cat([self._features(images[i:i + bs]) for i in range(0, len(images), bs)])

    # ---------- training ----------
    def fit(self, good_images: np.ndarray) -> None:
        f = self._batched(good_images)
        patches = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        self.bank = self._greedy_coreset(patches)

    def _greedy_coreset(self, patches: torch.Tensor) -> torch.Tensor:
        """Greedy k-center selection: keeps a small set of patches that still covers all of them."""
        n_keep = max(1, int(len(patches) * self.coreset_ratio))
        g = torch.Generator().manual_seed(self.seed)
        idx = [int(torch.randint(len(patches), (1,), generator=g))]
        dist = torch.cdist(patches, patches[idx]).squeeze(1)
        for _ in range(n_keep - 1):
            nxt = int(torch.argmax(dist))
            idx.append(nxt)
            dist = torch.minimum(dist, torch.cdist(patches, patches[nxt:nxt + 1]).squeeze(1))
        return patches[idx]

    # ---------- scoring ----------
    @torch.no_grad()
    def predict(self, images: np.ndarray, out_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (image scores N, heatmaps N x out_size x out_size)."""
        assert self.bank is not None, "fit first"
        f = self._batched(images)
        n, d, h, w = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(-1, d)
        dmin = torch.cat([torch.cdist(patches[i:i + 4096], self.bank).min(dim=1).values
                          for i in range(0, len(patches), 4096)])
        maps = dmin.view(n, 1, h, w)
        maps = F.interpolate(maps, size=(out_size, out_size), mode="bilinear", align_corners=False)
        maps = np.stack([gaussian_filter(m, 4) for m in maps.squeeze(1).numpy()])
        scores = maps.reshape(n, -1).max(axis=1)
        return scores, maps

    def save(self, path: str | Path) -> None:
        torch.save({"bank": self.bank, "proj": self.proj, "image_size": self.image_size,
                    "pretrained": self.pretrained}, path)
