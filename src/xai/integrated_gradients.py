"""
Integrated Gradients for CNN-LSTM + Temporal Attention.

Outputs:
  ig_variable_importance.png  — bar chart: mean |attr| per variable
  ig_spatial_<var>.png        — spatial map per variable (mean |attr| over time)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


def _integrated_gradients(lightning_module, x_spatial, x_temporal, n_steps=50):
    """
    Returns attributions: (window, n_vars, lat, lon) on CPU.
    """
    baseline = torch.zeros_like(x_spatial)
    grads = []

    for step in range(n_steps):
        alpha = step / (n_steps - 1)
        interp = (baseline + alpha * (x_spatial - baseline)).requires_grad_(True)
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        pred, _ = lightning_module.model.forward_with_attention(
            interp.float(), x_temporal.float()
        )
        pred.squeeze().backward()
        torch.backends.cudnn.enabled = prev
        grads.append(
            interp.grad.detach().squeeze(0).clone()
        )  # (window, n_vars, lat, lon)

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (x_spatial.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.cpu()


def _integrated_gradients_forward(model, x_spatial, x_temporal, n_steps=50):
    """
    Signed IG using raw model.forward() — for lstm_only (no attention).
    model: CNNLSTMModel (not LightningModule).
    Returns: (window, n_vars, lat, lon) on CPU.
    """
    baseline = torch.zeros_like(x_spatial)
    grads = []

    for step in range(n_steps):
        alpha = step / (n_steps - 1)
        interp = (baseline + alpha * (x_spatial - baseline)).requires_grad_(True)
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        pred = model(interp.float(), x_temporal.float())
        pred.squeeze().backward()
        torch.backends.cudnn.enabled = prev
        grads.append(interp.grad.detach().squeeze(0).clone())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (x_spatial.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.cpu()


def smoothgrad(
    lightning_module,
    x_spatial,
    x_temporal,
    n_steps=50,
    n_samples=20,
    sigma=0.1,
):
    """
    SmoothGrad: average IG over multiple noise-augmented copies of the input.

    Reduces high-frequency artefacts (e.g. the 8 px periodic grid from
    MaxPool2d) while preserving spatial structure.

    Args:
        lightning_module: CNNLightningModule
        x_spatial:   (1, window, C, H, W)
        x_temporal:  (1, window, 3)
        n_steps:     IG integration steps
        n_samples:   number of noisy runs to average
        sigma:       noise std as fraction of input range (x.max - x.min)
    Returns:
        attributions: (window, C, H, W) on CPU
    """
    x_range = x_spatial.max() - x_spatial.min()
    noise_std = sigma * x_range.item()

    acc = torch.zeros_like(x_spatial.squeeze(0))  # (window, C, H, W)
    for i in range(n_samples):
        noisy = (x_spatial + torch.randn_like(x_spatial) * noise_std).to(
            x_spatial.device
        )
        attr = _integrated_gradients(
            lightning_module, noisy, x_temporal, n_steps=n_steps
        )
        acc += attr
    return acc / n_samples


def smoothgrad_forward(
    model,
    x_spatial,
    x_temporal,
    n_steps=50,
    n_samples=20,
    sigma=0.1,
):
    """
    SmoothGrad using raw model.forward() — for lstm_only (no attention).
    model: CNNLSTMModel (not LightningModule).
    """
    x_range = x_spatial.max() - x_spatial.min()
    noise_std = sigma * x_range.item()

    acc = torch.zeros_like(x_spatial.squeeze(0))
    for i in range(n_samples):
        noisy = (x_spatial + torch.randn_like(x_spatial) * noise_std).to(
            x_spatial.device
        )
        attr = _integrated_gradients_forward(model, noisy, x_temporal, n_steps=n_steps)
        acc += attr
    return acc / n_samples


def _integrated_gradients_full(lightning_module, x_spatial, x_temporal, n_steps=50):
    """
    Like _integrated_gradients but also attributes x_temporal (year_norm, month_sin, month_cos).

    Returns:
        attr_spatial:  (window, n_vars, lat, lon) — same as _integrated_gradients
        attr_temporal: (window, 3)               — [year_norm, month_sin, month_cos]
    """
    baseline_s = torch.zeros_like(x_spatial)
    baseline_t = torch.zeros_like(x_temporal)
    grads_s, grads_t = [], []

    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    for step in range(n_steps):
        alpha = step / (n_steps - 1)
        interp_s = (baseline_s + alpha * (x_spatial - baseline_s)).requires_grad_(True)
        interp_t = (baseline_t + alpha * (x_temporal - baseline_t)).requires_grad_(True)

        pred, _ = lightning_module.model.forward_with_attention(
            interp_s.float(), interp_t.float()
        )
        pred.squeeze().backward()

        grads_s.append(interp_s.grad.detach().squeeze(0).clone())
        grads_t.append(interp_t.grad.detach().squeeze(0).clone())

    torch.backends.cudnn.enabled = prev

    avg_grads_s = torch.stack(grads_s).mean(dim=0)
    avg_grads_t = torch.stack(grads_t).mean(dim=0)

    attr_spatial = (x_spatial.squeeze(0) - baseline_s.squeeze(0)) * avg_grads_s
    attr_temporal = (x_temporal.squeeze(0) - baseline_t.squeeze(0)) * avg_grads_t

    return attr_spatial.cpu(), attr_temporal.cpu()


def analyze_integrated_gradients(
    lightning_module, test_dataset, output_dir, n_samples, config, lat, lon
):
    device = next(lightning_module.parameters()).device
    variables = config["variables"]
    ocean_vars = set(config.get("ocean_variables", variables))
    output_dir = Path(output_dir)
    n_vars = len(variables)

    rng = np.random.default_rng(42)
    idx_list = rng.choice(
        len(test_dataset), size=min(n_samples, len(test_dataset)), replace=False
    )

    spatial_accum = np.zeros((n_vars, len(lat), len(lon)))
    global_accum = np.zeros(n_vars)
    count = 0

    for idx in idx_list:
        xs, xt, _ = test_dataset[int(idx)]
        attrs = _integrated_gradients(
            lightning_module,
            xs.unsqueeze(0).to(device),
            xt.unsqueeze(0).to(device),
        )
        abs_attrs = attrs.abs()
        spatial_accum += abs_attrs.mean(dim=0).numpy()  # mean over time
        global_accum += abs_attrs.mean(dim=(0, 2, 3)).numpy()  # mean over time+space
        count += 1
        print(f"  IG {count}/{len(idx_list)}", end="\r")

    print()
    spatial_accum /= count
    global_accum /= count

    # --- Bar chart ---
    colors = ["tomato" if v in ocean_vars else "steelblue" for v in variables]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(variables, global_accum, color=colors)
    ax.set_xlabel("Mean |attribution| (normalised units)")
    ax.set_title("Variable importance — Integrated Gradients")
    ax.legend(
        handles=[
            Patch(color="tomato", label="ocean variable"),
            Patch(color="steelblue", label="atmospheric variable"),
        ],
        fontsize=8,
    )
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
        im = ax.imshow(
            data, origin="lower", extent=extent, cmap="YlOrRd", aspect="auto"
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"IG spatial attribution — {var}")
        plt.colorbar(im, ax=ax, label="|attribution|")
        plt.tight_layout()
        plt.savefig(output_dir / f"ig_spatial_{var}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved ig_spatial_{var}.png")

    return {
        "global_importance": dict(zip(variables, global_accum)),
        "spatial_maps": spatial_accum,
    }
