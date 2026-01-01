import torch
import torch.nn as nn
import logging

# 你的日志习惯
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger("Diff_vs_FM")

class GenerativeMechanism:
    """
    Comparison between Diffusion and Flow Matching training logic.
    """
    def __init__(self, mode = "diffusion"):
        self.mode = mode
        logger.info(f"Initialized mechanism with mode: {mode}")

    def get_training_pair(self, x_0):
        """
        Constructs x_t and the target for loss calculation.
        Args:
            x_0: Clean data (Batch, Dim)
        Returns:
            x_t: Noisy input for model
            t: Time step
            target: Ground truth label (Noise or Velocity)
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 1. Sample random time t in [0, 1]
        t = torch.rand(batch_size, device = device).view(-1, 1, 1, 1)
        
        # 2. Sample pure noise x_1
        x_1 = torch.randn_like(x_0)
        
        if self.mode == "diffusion":
            # --- Diffusion Logic (Simplified VP) ---
            # x_t = sqrt(1-t)*x_0 + sqrt(t)*epsilon 
            # (Note: Actual DDPM uses alpha_bar schedule, this is a simplification for contrast)
            
            # Usually Diffusion uses a specialized schedule, but conceptually:
            # Let's use linear interpolation for variance to keep shapes comparable to FM visualization
            # But standard DDPM is distinct. Let's stick to standard DDPM conceptual target.
            
            # Standard: x_t = mean * x_0 + std * noise
            # Target is usually the NOISE epsilon.
            
            # Using simple Linear Noise Schedule for demo (not strictly DDPM but conceptually aligned)
            alpha = 1 - t
            x_t = torch.sqrt(alpha) * x_0 + torch.sqrt(1 - alpha) * x_1
            
            target = x_1 # We predict the noise epsilon
            
        elif self.mode == "flow_matching":
            # --- Flow Matching Logic (Optimal Transport / Rectified Flow) ---
            # Path is a STRAIGHT LINE: x_t = (1 - t) * x_0 + t * x_1
            
            x_t = (1 - t) * x_0 + t * x_1
            
            # The velocity vector field u_t(x|x_0, x_1) = x_1 - x_0
            # This is constant velocity!
            target = x_1 - x_0 
            
        return x_t, t.squeeze(), target

# -----------------------------------------------------------
# 模拟训练循环中的一步
# -----------------------------------------------------------
if __name__ == "__main__":
    # Simulate specific inputs
    dummy_img = torch.randn(4, 3, 32, 32)
    
    # 1. Diffusion Way
    diff_mech = GenerativeMechanism(mode = "diffusion")
    x_t_diff, t_diff, target_diff = diff_mech.get_training_pair(dummy_img)
    
    logger.info(f"Diffusion Target (Noise) Mean: {target_diff.mean().item():.4f}")
    
    # 2. Flow Matching Way
    fm_mech = GenerativeMechanism(mode = "flow_matching")
    x_t_fm, t_fm, target_fm = fm_mech.get_training_pair(dummy_img)
    
    logger.info(f"Flow Matching Target (Velocity) Mean: {target_fm.mean().item():.4f}")
    
    # Key Takeaway for printing
    print("\n--- Key Difference ---")
    print(f"Diffusion Target Shape: {target_diff.shape} (Predicts pure noise)")
    print(f"Flow Matching Target Shape: {target_fm.shape} (Predicts vector x1 - x0)")