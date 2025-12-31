import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("CrossAttention_Debug")

class CrossAttention(nn.Module):
    """
    Corrected Cross-Attention Module with detailed logging.
    Implements Q (Visual) attending to K, V (Text/Context).
    """
    def __init__(
        self, 
        query_dim, 
        context_dim, 
        heads = 8, 
        dim_head = 64
    ):
        """
        Initialize the attention layer.
        
        Args:
            query_dim: Input dimension of the query (e.g., image features).
            context_dim: Input dimension of the context (e.g., text features).
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head # Saved for view()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        logger.info(f"Init CrossAttention: Heads={heads}, DimHead={dim_head}, InnerDim={inner_dim}")

        # Q projection (usually from Image)
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        
        # K, V projection (usually from Text/Context)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)
        )

    def forward(self, x, context = None):
        """
        Forward pass with shape logging.
        
        Args:
            x: Query tensor (Batch, Seq_len_Q, Dim_Q)
            context: Key/Value tensor (Batch, Seq_len_K, Dim_K)
        """
        b, n, d = x.shape
        logger.info(f"[Step 1] Input x shape: {x.shape}")

        # Use x as context if none provided (Self-Attention)
        if context is None:
            context = x
            logger.info("[Step 1.5] Context is None, using Self-Attention")
        else:
            logger.info(f"[Step 1.5] Context (Text) shape: {context.shape}")

        # 1. Projection and Reshape of Q
        # We project x to inner_dim, then split inner_dim into (heads, dim_head)
        # view(b, -1, heads, dim_head): -1 infers the sequence length 'n'
        q = self.to_q(x).view(b, -1, self.heads, self.dim_head)
        q = q.permute(0, 2, 1, 3) # (Batch, Heads, Seq_len_Q, Dim_Head)
        logger.info(f"[Step 2] Q reshaped & permuted: {q.shape}")

        # 2. Projection and Reshape of K, V
        # view(b, -1, heads, dim_head): -1 infers the context sequence length
        k = self.to_k(context).view(b, -1, self.heads, self.dim_head)
        k = k.permute(0, 2, 1, 3) # (Batch, Heads, Seq_len_K, Dim_Head)
        
        v = self.to_v(context).view(b, -1, self.heads, self.dim_head)
        v = v.permute(0, 2, 1, 3) # (Batch, Heads, Seq_len_K, Dim_Head)
        logger.info(f"[Step 3] K, V reshaped & permuted: {k.shape}")

        # 3. Attention Score Calculation
        # Q * K^T -> (Batch, Heads, Seq_len_Q, Seq_len_K)
        # This map tells us "how much each image pixel cares about each text token"
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim = -1)
        logger.info(f"[Step 4] Attention Map shape: {attn.shape}")

        # 4. Aggregation
        # Attn * V -> (Batch, Heads, Seq_len_Q, Dim_Head)
        out = torch.matmul(attn, v)
        logger.info(f"[Step 5] Weighted V shape: {out.shape}")

        # 5. Recombine Heads
        # Permute back: (Batch, Seq_len_Q, Heads, Dim_Head)
        out = out.permute(0, 2, 1, 3).contiguous()
        # Flatten heads: (Batch, Seq_len_Q, Inner_Dim)
        out = out.view(b, -1, self.heads * self.dim_head)
        logger.info(f"[Step 6] Output after merge heads: {out.shape}")
        
        # 6. Final Projection
        out = self.to_out(out)
        logger.info(f"[Step 7] Final Output shape: {out.shape}")
        
        return out

if __name__ == "__main__":
    # Main Function Logging Configuration
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )
    
    # 模拟数据
    # Batch size
    B = 1
    
    # Visual (Image): 64 pixels (flattened 8x8 feature map), 320 channels
    img_seq_len = 64
    img_dim = 320 
    dummy_visual = torch.randn(B, img_seq_len, img_dim)
    
    # Context (Text): 77 tokens (CLIP standard), 768 dim
    text_seq_len = 77
    text_dim = 768
    dummy_text = torch.randn(B, text_seq_len, text_dim)

    logger.info(">>> Starting CrossAttention Test")
    
    # 初始化模型
    model = CrossAttention(
        query_dim = img_dim, 
        context_dim = text_dim,
        heads = 8,
        dim_head = 64
    )
    
    # 前向传播
    output = model(dummy_visual, context = dummy_text)
    
    logger.info(">>> Test Finished Successfully")