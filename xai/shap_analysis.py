import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import load_config, get_feature_names




def analyze_shap(model, train_loader, test_loader, output_dir, config_path="/p/project1/hai_1127/radin1/exprecursors/config.yaml"):
    """SHAP ANALYSIS
    model: trained PyTorch model
    train_loader: DataLoader for training data (used for background=reference)
    test_loader: DataLoader for test data (used for SHAP values)
    output_dir: directory to save SHAP plots
    """
    device = next(model.parameters()).device
    model.eval() #trained NN 
    
    background = []
    for i, batch in enumerate(train_loader):
        if i >= 20: break
        x, _ = batch
        if isinstance(x, dict):
            background.append(x['image'])
        else:
            background.append(x)
    
    background = torch.cat(background)[:50].to(device).float() 
    

    explainer = shap.GradientExplainer(model, background)

    test_samples = []
    for i, batch in enumerate(test_loader):
        if i >= 5: break
        x, _ = batch
        if isinstance(x, dict):
            test_samples.append(x['image'])
        else:
            test_samples.append(x)
    
    test_samples = torch.cat(test_samples)[:25].to(device)
    
    shap_values = explainer.shap_values(test_samples)

    config = load_config(config_path)
    feature_names = get_feature_names(config)

    
    shap_values_reshaped = np.array(shap_values).reshape(len(feature_names), -1)
    test_samples_reshaped = test_samples.cpu().numpy().reshape(len(feature_names), -1)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_reshaped.T, 
        test_samples_reshaped.T,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_reshaped.T,
        test_samples_reshaped.T,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
   
    return {
        'shap_values': shap_values,
        'feature_names': feature_names
    }