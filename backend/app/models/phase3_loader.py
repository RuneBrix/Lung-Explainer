from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE3_CKPT = "../assets/checkpoints/model_phase3.pth"
CALIBRATION_JSON = "../assets/checkpoints/calibration_phase3.json"



class UNetEncoderClassifier(nn.Module):
    def __init__(self, in_ch=1, widths=(32, 64, 128, 256), norm_layer=nn.BatchNorm3d):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                norm_layer(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                norm_layer(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, widths[0])
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = block(widths[0], widths[1])
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = block(widths[1], widths[2])
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = block(widths[2], widths[3])
        self.head = nn.Linear(widths[3], 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        x = self.enc4(x)
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.head(x)

def _strip_prefixes(sd: dict, prefixes=("_orig_mod.", "module.")) -> dict:
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        new_sd[k] = v
    return new_sd

def _to_cuda_safe(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    if not torch.cuda.is_available():
        return model
    use_cl3d = os.environ.get("USE_CHANNELS_LAST_3D", "1") == "1"
    mf = getattr(torch, "channels_last_3d", None)
    if use_cl3d and mf is not None:
        try:
            return model.to("cuda", memory_format=mf)
        except (RuntimeError, AttributeError) as e:
            print("[load_phase3_model] channels_last_3d failed, falling back:", e)
            return model.to("cuda")
    else:
        return model.to("cuda")
    
def _detect_arch(sd: dict) -> str:
    has_dense = any(k.startswith("features.") for k in sd.keys())  # MONAI DenseNet
    has_unet = any(k.startswith("enc1.") for k in sd.keys())
    if has_dense and not has_unet:
        return "densenet121"
    if has_unet and not has_dense:
        return "unetenc"
    return "densenet121"  # fallback

def load_phase3_model():
    ckpt = Path(PHASE3_CKPT)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["state_dict"] if (isinstance(obj, dict) and "state_dict" in obj) else obj

    if not isinstance(state_dict, dict):
        model = obj
        model = _to_cuda_safe(model)
        print(f"[load_phase3_model] Loaded model object from {ckpt}")
        return model

    state_dict = _strip_prefixes(state_dict)
    arch = _detect_arch(state_dict)

    if arch == "densenet121":
        from monai.networks.nets import DenseNet121

        model = DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.0
        )
        arch_name = "DenseNet121-3D (MONAI)"
    else:
        model = UNetEncoderClassifier(in_ch=1, widths=(32, 64, 128, 256))
        arch_name = "UNetEncoderClassifier-3D"

    try:
        result = model.load_state_dict(state_dict, strict=True)
        mk = getattr(result, "missing_keys", [])
        uk = getattr(result, "unexpected_keys", [])
        if mk or uk:
            print(
                f"[load_phase3_model] strict=True reported Missing: {mk} | Unexpected: {uk}"
            )
    except RuntimeError as e:
        print(f"[load_phase3_model] strict=True failed ({e}); retrying with strict=False.")
        result = model.load_state_dict(state_dict, strict=False)
        mk = getattr(result, "missing_keys", [])
        uk = getattr(result, "unexpected_keys", [])
        print(
            f"[load_phase3_model] Loaded with strict=False. Missing: {mk} | Unexpected: {uk}"
        )

    model = _to_cuda_safe(model)
    print(f"[load_phase3_model] Loaded {arch_name} from {ckpt}")
    return model