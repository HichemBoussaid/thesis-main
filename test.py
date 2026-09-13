import torch
import numpy as np
from scipy.stats import pearsonr
gt = torch.load("/lustre/fsn1/projects/rech/dvj/unr53mf/DatasetParis/recon_qual_check/SAMPLE_S2-mono_gt.pt").numpy()       # (10, 100, 100)
recon = torch.load("/lustre/fsn1/projects/rech/dvj/unr53mf/DatasetParis/recon_qual_check/SAMPLE_S2-mono_recon.pt").numpy()  # (10, 100, 100)
for c in range(gt.shape[0]):
    corr, _ = pearsonr(gt[c].flatten(), recon[c].flatten())

    print(f"band {c}: correlation = {corr:.3f}, "
    f"gt range=[{gt[c].min():.2f},{gt[c].max():.2f}], "
    f"recon range=[{recon[c].min():.2f},{recon[c].max():.2f}]")