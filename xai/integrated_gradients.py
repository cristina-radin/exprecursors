"""
Integrated Gradients for CNN-LSTM + Temporal Attention.

Outputs:
  ig_variable_importance.png  — bar chart: mean |attr| per variable
  ig_spatial_<var>.png        — spatial map per variable (mean |attr| over time)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


def _integrated_gradients(lightning_module, x_spatial, x_temporal, n_steps=50):
    """
    Returns attributions: (window, n_vars, lat, lon) on CPU.
    """
    baseline = torch.zeros_like(x_spatial)
    grads    = []

    for step in range(n_steps):
        alpha  = step / (n_steps - 1)
        interp = (baseline + alpha * (x_spatial - baseline)).requires_grad_(True)
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        pred, _ = lightning_module.model.forward_with_attention(
            interp.float(), x_temporal.float()
        )
        pred.squeeze().backward()
        torch.backends.cudnn.enabled = prev
        grads.append(interp.grad.detach().squeeze(0).clone())   # (window, n_vars, lat, lon)

    avg_grads    = torch.stack(grads).mean(dim=0)
    attributions = (x_spatial.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.cpu()


def analyze_integrated_gradients(lightning_module, test_dataset, output_dir,
                                  n_samples, config, lat, lon):
    device     = next(lightning_module.parameters()).device
    variables  = config["variables"]
    ocean_vars = set(config.get("ocean_variables", variables))
    output_dir = Path(output_dir)
    n_vars     = len(variables)

    rng      = np.random.default_rng(42)
    idx_list = rng.choice(len(test_dataset),
                          size=min(n_samples, len(test_dataset)),
                          replace=False)

    spatial_accum = np.zeros((n_vars, len(lat), len(lon)))
    global_accum  = np.zeros(n_vars)
    count         = 0

    for idx in idx_list:
        xs, xt, _ = test_dataset[int(idx)]
        attrs = _integrated_gradients(
            lightning_module,
            xs.unsqueeze(0).to(device),
            xt.unsqueeze(0).to(device),
        )
        abs_attrs       = attrs.abs()
        spatial_accum  += abs_attrs.mean(dim=0).numpy()       # mean over time
        global_accum   += abs_attrs.mean(dim=(0, 2, 3)).numpy()  # mean over time+space
        count          += 1
        print(f"  IG {count}/{len(idx_list)}", end="\r")

    print()
    spatial_accum /= count
    global_accum  /= count

    # --- Bar chart ---
    colors = ["tomato" if v in ocean_vars else "steelblue" for v in variables]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(variables, global_accum, color=colors)
    ax.set_xlabel("Mean |attribution| (normalised units)")
    ax.set_title("Variable importance — Integrated Gradients")
    ax.legend(handles=[
        Patch(color="tomato",    label="ocean variable"),
        Patch(color="steelblue", label="atmospheric variable"),
    ], fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "ig_variable_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ig_variable_importance.png")

    # --- Spatial maps ---
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    for i, var in enumerate(variables):
        data = spatial_accum[i].copy()
        if var in ocean_vars:
            data = np.where(data == 0, np.nan, data)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(f"IG spatial attribution — {var}")
        plt.colorbar(im, ax=ax, label="|attribution|")
        plt.tight_layout()
        plt.savefig(output_dir / f"ig_spatial_{var}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved ig_spatial_{var}.png")

    return {
        "global_importance": dict(zip(variables, global_accum)),
        "spatial_maps":      spatial_accum,
    }
