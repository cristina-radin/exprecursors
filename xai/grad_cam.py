import torch
from torch import device
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import load_config, get_feature_names


class GradCAM:
    """Grad-CAM implementation for PyTorch models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x, class_idx=None):
        output = self.model(x)
        
        if class_idx is None:
            if output.shape[1] == 1:
                class_idx = 0
            else:
                class_idx = output.argmax().item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


def analyze_gradcam(model, dataset, output_dir, num_samples=5, config_path="/p/project1/hai_1127/radin1/exprecursors/config.yaml"):
    device = next(model.parameters()).device
    model.eval()
    
    config = load_config(config_path)
    feature_names = get_feature_names(config)
    
    # land indexs
    land_mask_idx = None
    physical_vars = []
    temporal_vars = []
    
    for i, name in enumerate(feature_names):
        if name == 'land_mask':
            land_mask_idx = i
        elif name in ['year', 'month_sin', 'month_cos']:
            temporal_vars.append(name)
        else:
            physical_vars.append(name)
    
    
    mhw_indices = []
    normal_indices = []
    
    for i in range(min(100, len(dataset))):
        _, y = dataset[i]
        if y.item() == 1 and len(mhw_indices) < num_samples:
            mhw_indices.append(i)
        elif y.item() == 0 and len(normal_indices) < num_samples:
            normal_indices.append(i)
        
        if len(mhw_indices) >= num_samples and len(normal_indices) >= num_samples:
            break
    
    # Last conv layer
    target_layer = None
    for module in model.model.network.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        print("Conv layer not found, using default target layer.")
        target_layer = model.model.network[-6]
    
    results = {}
    
    for category, indices in [('MHW', mhw_indices), ('noMHW', normal_indices)]:
        results[category] = []
        for idx in indices:
            x, y = dataset[idx]  # x shape: [channels, lat, lon]
            x_batch = x.unsqueeze(0).to(device).float()
            
            with torch.no_grad():
                logits = model(x_batch)
                prob = torch.sigmoid(logits).item()
            
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate(x_batch)
            

            # Normalized gradcam for visualization
            cam_normalized = cam.copy()
            cam_normalized = cam_normalized / cam_normalized.max()  


            if land_mask_idx is not None:
                land_mask = x[land_mask_idx].cpu().numpy()  # 0=land, 1=ocean
                cam = cam * land_mask  
                
                land_pixels = (land_mask == 0).sum()
                if land_pixels > 0 and 'zos' in feature_names:
                    zos_idx = feature_names.index('zos')
            else:
                print("Land_mask not founded")
            
            # PLOT
            fig, axes = plt.subplots(2, len(physical_vars), figsize=(5*len(physical_vars), 8))

            if len(physical_vars) == 1:
                axes = axes.reshape(2, 1)

            cmap_physical = plt.cm.RdBu_r.copy()
            cmap_physical.set_bad("lightgray")  

            cmap_cam = plt.cm.jet.copy()
            cmap_cam.set_bad("lightgray")  

            for i, var_name in enumerate(physical_vars):
                var_idx = feature_names.index(var_name)
                
                var_data = x[var_idx].cpu().numpy()
                land_mask = x[land_mask_idx].cpu().numpy()
                var_data_with_nan = var_data.copy()
                var_data_with_nan[land_mask == 0] = np.nan
                
                im1 = axes[0, i].imshow(var_data_with_nan, cmap=cmap_physical, origin='lower')
                axes[0, i].set_title(f'Input: {var_name}')
                plt.colorbar(im1, ax=axes[0, i])
                
                cam_with_nan = cam_normalized.copy()
                cam_with_nan[land_mask == 0] = np.nan
                im2 = axes[1, i].imshow(cam_with_nan, cmap=cmap_cam, alpha=0.7, origin='lower', vmin=0, vmax=1)
                #axes[1, i].imshow(var_data_with_nan, cmap=cmap_physical, alpha=0.7, origin='lower')
                axes[1, i].set_title(f'Grad-CAM {var_name}')
                plt.colorbar(im2, ax=axes[1, i])

            plt.suptitle(f'{category} - index {idx} - Prob: {prob:.3f} - Real: {y.item()}')
            plt.tight_layout()

            filename = Path(output_dir) / f'gradcam_{category}_idx{idx}_prob{prob:.2f}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
                        
            results[category].append({
                'idx': idx,
                'prob': prob,
                'true': y.item(),
                'cam': cam,
                'file': filename
            })
            
            print(f"{category} {idx}: prob={prob:.3f}, true={y.item()}")
    
    return results