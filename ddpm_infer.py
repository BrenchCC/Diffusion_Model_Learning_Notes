import os
import argparse
import logging

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Local import
from models.unet_model import SimpleUNet

# Global Logger
logger = logging.getLogger("Inference_Demo")

def load_image_for_inference(path, size, device):
    """Load image as GT reference"""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0).to(device)

def tensor_to_numpy(tensor):
    """Convert tensor [-1, 1] to numpy [0, 1] for plotting"""
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    return tensor.permute(1, 2, 0).numpy()

def get_noisy_input(x_start, t, device):
    """Generate a noisy input on the fly"""
    noise = torch.randn_like(x_start)
    w = t.view(-1, 1, 1, 1).float() / 1000.0
    w = w * 0.8
    return (1 - w) * x_start + w * noise

def args_parser():
    parser = argparse.ArgumentParser(description = "DDPM Inference Demo")
    parser.add_argument("--img_size", type = int, default = 256, help = "Image size")
    parser.add_argument("--image_path", type = str, default = "images/demo.png", help = "Path to image")
    parser.add_argument("--model_path", type = str, default = "model_ckpt/demo_unet.pth", help = "Path to model")
    parser.add_argument("--result_path", type = str, default = "images/inference_result.png", help = "Path to save result")
    return parser.parse_args()

if __name__ == "__main__":
    # 1. Logging Setup
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )
    
    # 2. Config
    args = args_parser()
    IMG_SIZE = args.img_size
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    MODEL_PATH = args.model_path
    IMAGE_PATH = args.image_path
    RESULT_PATH = args.result_path
    
    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. Please run train_demo.py first.")
        exit(1)
        
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = SimpleUNet(
        image_channels = 3, 
        down_channels = (64, 128, 256), # Must match training config
        up_channels = (256, 128, 64), 
        time_emb_dim = 128
    ).to(DEVICE)
    
    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location = DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully.")
    
    # 4. Inference
    logger.info("Processing image...")
    
    # Load GT for comparison
    gt_img = load_image_for_inference(IMAGE_PATH, IMG_SIZE, DEVICE)
    
    # Create a random noisy input (Simulating t=750)
    inference_t = torch.tensor([500], device = DEVICE).long()
    noisy_input = get_noisy_input(gt_img, inference_t, DEVICE)
    
    with torch.no_grad():
        denoised_output = model(noisy_input, inference_t)
        
    # 5. Visualization
    logger.info(f"Saving visualization to {RESULT_PATH}")
    
    fig, axes = plt.subplots(1, 3, figsize = (20, 10))
    
    # Plot GT
    axes[0].imshow(tensor_to_numpy(gt_img))
    axes[0].set_title("Ground Truth", fontsize = 14)
    axes[0].axis("off")
    
    # Plot Input
    axes[1].imshow(tensor_to_numpy(noisy_input))
    axes[1].set_title(f"Noisy Input (t={inference_t.item()})", fontsize = 14)
    axes[1].axis("off")
    
    # Plot Output
    axes[2].imshow(tensor_to_numpy(denoised_output))
    axes[2].set_title("Inference Output (Denoised)", fontsize = 14)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi = 400)
    plt.close()
    
    logger.info("Inference complete.")