"""
Evaluation and visualization script for CycleGAN SAR to EO conversion.
Generates results, visualizations, and metrics for model assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from tqdm.auto import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from config import *
from preprocess import create_dataloaders, setup_data_environment
from models import create_models
from train_cycleGAN import load_checkpoint


def denormalize_image(tensor, sensor_type='s2'):
    """
    Convert normalized tensor back to original range.
    
    Args:
        tensor: Normalized tensor in range [-1, 1]
        sensor_type: 's1' for SAR or 's2' for EO
        
    Returns:
        Denormalized tensor
    """
    if sensor_type == 's2':
        # EO denormalization
        tensor = (tensor + 1) / 2.0  # Convert to [0, 1]
        tensor = tensor * (EO_NORM_RANGE[1] - EO_NORM_RANGE[0]) + EO_NORM_RANGE[0]
    elif sensor_type == 's1':
        # SAR denormalization
        tensor = (tensor + 1) / 2.0  # Convert to [0, 1]
        tensor = tensor * (abs(SAR_NORM_RANGE[0]) - SAR_NORM_RANGE[1]) - abs(SAR_NORM_RANGE[0])
    
    return tensor


def tensor_to_display_format(tensor, band_indices):
    """
    Convert tensor to format suitable for display.
    
    Args:
        tensor: Input tensor
        band_indices: List of band indices to extract
        
    Returns:
        Numpy array ready for visualization
    """
    # Extract specified bands
    if len(band_indices) == 3:
        img = tensor[band_indices].cpu().detach().numpy()
    else:
        img = tensor[band_indices[0]].cpu().detach().numpy()
        img = np.expand_dims(img, axis=0)
    
    # Convert to display range [0, 1]
    img = (img + 1) / 2.0
    img = np.clip(img, 0, 1)
    
    # Transpose for matplotlib (H, W, C)
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    
    return img


def visualize_generated_image(generated_tensor, ground_truth_tensor, 
                            save_path=None, show_plot=True):
    """
    Create visualization comparing generated and ground truth images.
    
    Args:
        generated_tensor: Generated EO image tensor
        ground_truth_tensor: Ground truth EO image tensor
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.suptitle("Model Output vs. Ground Truth", fontsize=16)

    # RGB composite
    gen_rgb = tensor_to_display_format(generated_tensor, RGB_INDICES)
    gt_rgb = tensor_to_display_format(ground_truth_tensor, RGB_INDICES)
    
    # NIR-SWIR-RE composite
    gen_nir = tensor_to_display_format(generated_tensor, NIR_SWIR_RE_INDICES)
    gt_nir = tensor_to_display_format(ground_truth_tensor, NIR_SWIR_RE_INDICES)

    # Plot images
    axes[0, 0].imshow(gen_rgb)
    axes[0, 0].set_title("Generated - RGB")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gen_nir)
    axes[0, 1].set_title("Generated - NIR SWIR RE")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(gt_rgb)
    axes[1, 0].set_title("Ground Truth - RGB")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt_nir)
    axes[1, 1].set_title("Ground Truth - NIR SWIR RE")
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_individual_images(generated_tensor, ground_truth_tensor, 
                         sample_idx, output_dir=GENERATED_SAMPLES_DIR):
    """
    Save individual generated and ground truth images.
    
    Args:
        generated_tensor: Generated EO image tensor
        ground_truth_tensor: Ground truth EO image tensor
        sample_idx: Sample index for naming
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert RGB bands to PIL Images
    gen_rgb = tensor_to_display_format(generated_tensor, RGB_INDICES)
    gt_rgb = tensor_to_display_format(ground_truth_tensor, RGB_INDICES)
    
    # Save as PNG files
    gen_img = Image.fromarray((gen_rgb * 255).astype(np.uint8))
    gt_img = Image.fromarray((gt_rgb * 255).astype(np.uint8))
    
    gen_path = os.path.join(output_dir, f'generated_eo_{sample_idx}.png')
    gt_path = os.path.join(output_dir, f'real_eo_{sample_idx}.png')
    
    gen_img.save(gen_path)
    gt_img.save(gt_path)
    
    print(f"Saved: {gen_path} and {gt_path}")


def calculate_metrics(generated_tensor, ground_truth_tensor):
    """
    Calculate image quality metrics between generated and ground truth images.
    
    Args:
        generated_tensor: Generated image tensor
        ground_truth_tensor: Ground truth image tensor
        
    Returns:
        Dictionary of metrics
    """
    device = generated_tensor.device
    
    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    # Convert to [0, 1] range for metrics
    gen_norm = (generated_tensor + 1.0) / 2.0
    gt_norm = (ground_truth_tensor + 1.0) / 2.0
    
    metrics = {}
    
    # Calculate metrics for RGB bands
    gen_rgb = gen_norm[RGB_INDICES].unsqueeze(0)
    gt_rgb = gt_norm[RGB_INDICES].unsqueeze(0)
    
    metrics['ssim_rgb'] = ssim(gen_rgb, gt_rgb).item()
    metrics['psnr_rgb'] = psnr(gen_rgb, gt_rgb).item()
    
    # Calculate metrics for NIR-SWIR-RE bands
    gen_nir = gen_norm[NIR_SWIR_RE_INDICES].unsqueeze(0)
    gt_nir = gt_norm[NIR_SWIR_RE_INDICES].unsqueeze(0)
    
    metrics['ssim_nir'] = ssim(gen_nir, gt_nir).item()
    metrics['psnr_nir'] = psnr(gen_nir, gt_nir).item()
    
    # Calculate metrics for all bands
    gen_all = gen_norm.unsqueeze(0)
    gt_all = gt_norm.unsqueeze(0)
    
    metrics['ssim_all'] = ssim(gen_all, gt_all).item()
    metrics['psnr_all'] = psnr(gen_all, gt_all).item()
    
    # Calculate L1 loss
    metrics['l1_loss'] = F.l1_loss(generated_tensor, ground_truth_tensor).item()
    
    return metrics


def evaluate_model(model, dataloader, device=DEVICE, num_samples=10):
    """
    Evaluate model on a dataset and calculate average metrics.
    
    Args:
        model: Generator model to evaluate
        dataloader: Data loader for evaluation
        device: Device for computation
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i, (sar_batch, eo_batch) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= num_samples:
                break
                
            sar_batch = sar_batch.to(device)
            eo_batch = eo_batch.to(device)
            
            # Generate EO images from SAR
            generated_eo = model(sar_batch)
            
            # Calculate metrics for each sample in batch
            for j in range(sar_batch.size(0)):
                metrics = calculate_metrics(generated_eo[j], eo_batch[j])
                all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[key + '_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics


def generate_sample_results(checkpoint_path, dataloader, num_samples=5, 
                          output_dir=GENERATED_SAMPLES_DIR):
    """
    Generate sample results from a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: Data loader for samples
        num_samples: Number of samples to generate
        output_dir: Output directory for results
    """
    print("Loading model and checkpoint...")
    
    # Create models
    sar_to_eo, eo_to_sar, sar_critic, eo_critic = create_models()
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, sar_to_eo, eo_to_sar, sar_critic, eo_critic)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Set to evaluation mode
    sar_to_eo.eval()
    
    print(f"Generating {num_samples} sample results...")
    
    with torch.no_grad():
        for i, (sar_batch, eo_batch) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            sar_batch = sar_batch.to(DEVICE)
            eo_batch = eo_batch.to(DEVICE)
            
            # Generate EO images
            generated_eo = sar_to_eo(sar_batch)
            
            # Process each sample in the batch
            for j in range(min(sar_batch.size(0), num_samples - i)):
                sample_idx = i * dataloader.batch_size + j + 1
                
                # Create visualization
                viz_path = os.path.join(output_dir, f'comparison_{sample_idx}.png')
                visualize_generated_image(
                    generated_eo[j], eo_batch[j], 
                    save_path=viz_path, show_plot=False
                )
                
                # Save individual images
                save_individual_images(generated_eo[j], eo_batch[j], sample_idx, output_dir)
                
                # Calculate and print metrics
                metrics = calculate_metrics(generated_eo[j], eo_batch[j])
                print(f"Sample {sample_idx} metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                print()
    
    print(f"Sample results saved to: {output_dir}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN SAR to EO model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=GENERATED_SAMPLES_DIR,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Setting up evaluation environment...")
    setup_data_environment()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(args.data_path)
    
    # Use validation loader for evaluation
    eval_loader = val_loader if val_loader is not None else train_loader
    
    # Generate sample results
    generate_sample_results(
        checkpoint_path=args.checkpoint,
        dataloader=eval_loader,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # Evaluate model performance
    print("Evaluating model performance...")
    sar_to_eo, _, _, _ = create_models()
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        sar_to_eo.load_state_dict(checkpoint['sar_to_eo'])
        
        avg_metrics = evaluate_model(sar_to_eo, eval_loader, num_samples=args.num_samples)
        
        print("\nAverage Metrics:")
        print("-" * 40)
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
    
    print("Evaluation completed!")


if __name__ == "__main__":
    # For direct usage without command line arguments
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_5.pt")  # Adjust as needed
    
    if len(os.sys.argv) == 1:  # No command line arguments
        setup_data_environment()
        _, val_loader = create_dataloaders(DATA_PATH)
        
        if os.path.exists(checkpoint_path):
            generate_sample_results(checkpoint_path, val_loader)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Please provide a valid checkpoint path")
    else:
        main()
