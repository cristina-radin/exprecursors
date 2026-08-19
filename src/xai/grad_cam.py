"""
Attention-weighted Grad-CAM for CNN-LSTM + Temporal Attention.

Strategy:
  1. Hook the last Conv2d of the CNN encoder (before AdaptiveAvgPool).
  2. Forward: collect CNN activations for all 60 frames + attention weights.
  3. Backward: collect CNN gradients.
  4. For each timestep t: cam_t = ReLU( mean_c(grad_t * act_t) )
  5. Final map = sum_t( attn_t * cam_t )  →  upsampled to (lat, lon).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Grad-CAM engine
# ---------------------------------------------------------------------------


class AttentionGradCAM:
    def __init__(self, lightning_module):
        self.lm = lightning_module
        self.model = lightning_module.model
        self._acts = None
        self._grads = None
        # arch="lstm_only" never builds self.attention (see cnn_lstm.py —
        # only arch != "lstm_only" creates TemporalAttention), so
        # forward_with_attention() would raise AttributeError on
        # self.attention.attn(...). Branch instead of crashing.
        self.is_lstm_only = getattr(self.model, "attention", None) is None

        target_layer = self._find_last_conv()
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _find_last_conv(self):
        last = None
        for m in self.model.cnn_encoder.cnn:
            if isinstance(m, torch.nn.Conv2d):
                last = m
        assert last is not None, "No Conv2d found in cnn_encoder"
        return last

    def _save_act(self, module, inp, out):
        self._acts = out  # (batch*window, C, H', W')

    def _save_grad(self, module, grad_in, grad_out):
        self._grads = grad_out[0]  # (batch*window, C, H', W')

    def compute(self, x_spatial, x_temporal):
        """
        Returns:
            cam:          (lat, lon)  — spatial importance [0, 1]
            attn_weights: (window,) softmax attention weights, or None for
                arch="lstm_only" (no attention module exists — see __init__).
        """
        # cuDNN LSTM requires the same backend for forward+backward.
        # In eval mode backward is not supported, so disable cuDNN for both.
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        self.model.zero_grad()
        if self.is_lstm_only:
            pred = self.model(x_spatial.float(), x_temporal.float())
            attn = None
        else:
            pred, attn = self.model.forward_with_attention(
                x_spatial.float(), x_temporal.float()
            )
        pred.squeeze().backward()
        torch.backends.cudnn.enabled = prev

        window = x_spatial.shape[1]
        lat = x_spatial.shape[3]
        lon = x_spatial.shape[4]

        acts = self._acts.detach()  # (window, C, H', W')
        grads = self._grads.detach()

        cam = torch.zeros(acts.shape[2], acts.shape[3], device=acts.device)
        if self.is_lstm_only:
            # No attention score exists to weight cam_t by. The LSTM's own
            # backprop-through-time already attenuates grad_t for timesteps
            # it relied on less to form the last hidden state (the only one
            # the FC head sees, lstm_out[:, -1, :]) — so cam_t's own
            # magnitude already carries that weighting; sum unweighted.
            attn_w = None
            for t in range(window):
                w_t = grads[t].mean(dim=(1, 2), keepdim=True)  # (C,1,1)
                cam_t = F.relu((w_t * acts[t]).sum(dim=0))  # (H',W')
                cam = cam + cam_t
        else:
            attn_w = attn.squeeze(0).detach()  # (window,)
            for t in range(window):
                w_t = grads[t].mean(dim=(1, 2), keepdim=True)  # (C,1,1)
                cam_t = F.relu((w_t * acts[t]).sum(dim=0))  # (H',W')
                cam = cam + attn_w[t] * cam_t

        cam = cam / (cam.max() + 1e-8)
        cam = (
            F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(lat, lon),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return cam, (attn_w.cpu().numpy() if attn_w is not None else None)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_cam(cam, xs_np, variables, ocean_vars, lat, lon, title, save_path):
    """Row 0: last input frame per variable. Row 1: Grad-CAM."""
    n_vars = len(variables)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, axes = plt.subplots(2, n_vars, figsize=(4 * n_vars, 7))
    if n_vars == 1:
        axes = axes.reshape(2, 1)

    for i, var in enumerate(variables):
        frame = xs_np[-1, i].copy()  # last day of window, (lat, lon)
        if var in ocean_vars:
            frame = np.where(frame == 0, np.nan, frame)
        vmax = np.nanpercentile(np.abs(frame), 98) or 1.0

        ax0 = axes[0, i]
        im0 = ax0.imshow(
            frame,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax0.set_title(var, fontsize=9)
        ax0.set_xlabel("lon")
        ax0.set_ylabel("lat")
        plt.colorbar(im0, ax=ax0, fraction=0.03)

        ax1 = axes[1, i]
        im1 = ax1.imshow(
            cam,
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax1.set_title("Grad-CAM", fontsize=9)
        ax1.set_xlabel("lon")
        ax1.set_ylabel("lat")
        plt.colorbar(im1, ax=ax1, fraction=0.03)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_attention(attn_list, labels, save_path):
    window = len(attn_list[0])
    days = np.arange(-window + 1, 1)  # -59 … 0

    fig, ax = plt.subplots(figsize=(10, 4))
    for attn, label in zip(attn_list, labels):
        ax.plot(days, attn, alpha=0.7, label=label)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, label="last input day")
    ax.set_xlabel("Day relative to last input day")
    ax.set_ylabel("Attention weight")
    ax.set_title("Temporal attention weights")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_gradcam(
    lightning_module, test_dataset, output_dir, n_samples, config, lat, lon
):
    device = next(lightning_module.parameters()).device
    variables = config["variables"]
    ocean_vars = set(config.get("ocean_variables", variables))
    output_dir = Path(output_dir)

    engine = AttentionGradCAM(lightning_module)
    lightning_module.eval()

    # Collect predictions on the test set
    preds = []
    for idx in range(len(test_dataset)):
        xs, xt, _ = test_dataset[idx]
        with torch.no_grad():
            p, _ = lightning_module.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
        preds.append(p.item())
    preds = np.array(preds)
    order = np.argsort(preds)

    selected = {
        "high_anom": list(order[-n_samples:][::-1]),
        "low_anom": list(order[:n_samples]),
    }

    attn_all, labels_all = [], []

    for category, idx_list in selected.items():
        for rank, idx in enumerate(idx_list):
            xs, xt, y_true = test_dataset[idx]
            xs_dev = xs.unsqueeze(0).to(device)
            xt_dev = xt.unsqueeze(0).to(device)

            cam, attn = engine.compute(xs_dev, xt_dev)
            label = (
                f"{category} #{rank+1}  "
                f"pred={preds[idx]:.2f}  true={y_true.item():.2f}"
            )
            attn_all.append(attn)
            labels_all.append(label)

            save_path = output_dir / f"gradcam_{category}_{rank+1}.png"
            _plot_cam(
                cam, xs.numpy(), variables, ocean_vars, lat, lon, label, save_path
            )
            print(f"  Saved {save_path.name}")

    _plot_attention(attn_all, labels_all, output_dir / "attention_weights.png")
    print("  Saved attention_weights.png")
    return {"attn": attn_all, "labels": labels_all}
