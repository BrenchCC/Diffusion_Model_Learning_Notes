import os
import argparse
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models.unet_model import SimpleUNet

logger = logging.getLogger("Train_Demo")

def load_image(path, size, device):
    """Load and normalize image to [-1, 1]"""
    if not os.path.exists(path):
        logger.warning(f"Image {path} not found. Generating random noise image.")
        img = Image.fromarray(torch.randint(0, 255, (size, size, 3), dtype = torch.uint8).numpy())
        img.save(path)
    else:
        img = Image.open(path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0).to(device)

def get_noisy_image(x_start, t, noise):
    """Add noise based on timestep t"""
    w = t.view(-1, 1, 1, 1).float() / 1000.0
    w = w * 0.8 
    return (1 - w) * x_start + w * noise

def args_parser():
    parser = argparse.ArgumentParser(description = "Train Diffusion Model")
    parser.add_argument("--img_size", type = int, default = 256, help = "Image size")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--train_steps", type = int, default = 3000, help = "Training steps")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--image_path", type = str, default = "images/demo.png", help = "Path to image")
    parser.add_argument("--model_save_path", type = str, default = "model_ckpt/demo_unet.pth", help = "Path to save model")
    return parser.parse_args()

if __name__ == "__main__":
    # 1. Logging Setup
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )
    
    args = args_parser()
    
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    TRAIN_STEPS = args.train_steps
    LR = args.lr
    IMAGE_PATH = args.image_path
    MODEL_SAVE_PATH = args.model_save_path

    # 2. Config & Directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok = True)
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok = True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    
    logger.info(f"Using Device: {DEVICE}")

    # 3. Data & Model Init
    clean_img = load_image(IMAGE_PATH, IMG_SIZE, DEVICE)
    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LR)
    
    logger.info("Start Training...")
    
    # 4. Training Loop
    model.train()
    pbar = tqdm(range(TRAIN_STEPS), desc = "Training")
    
    for step in pbar:
        optimizer.zero_grad()
        
        t = torch.randint(0, 1000, (BATCH_SIZE,), device = DEVICE).long()
        noise = torch.randn_like(clean_img)
        noisy_img = get_noisy_image(clean_img, t, noise)
        
        predicted_img = model(noisy_img, t)
        
        loss = F.mse_loss(predicted_img, clean_img)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        
    # 5. Save Model
    logger.info(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Training finished and model saved.")