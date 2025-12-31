import logging

import torch
from torch import nn

logger = logging.getLogger("Conv2d_Demo")

def conv_demo(
    input_value: torch.Tensor,
    kernel_config: dict,
):
    """
    演示卷积层的前向传播
    """
    # 1. 提取配置参数
    in_channels = kernel_config["in_channels"]
    out_channels = kernel_config["out_channels"]
    kernel_size = kernel_config["kernel_size"]
    stride = kernel_config["stride"]
    padding = kernel_config["padding"]

    # 2. 创建卷积层
    conv_layer = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        bias = False,
    )

    # 3. 初始化卷积核（这里简单用随机数）
    conv_layer.weight.data.normal_(mean = 0.0, std = 0.02)
    
    # 4. 前向传播
    output = conv_layer(input_value)
    
    return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )

    test_config = {
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
    }
    
    # 6. 生成随机输入 (batch_size, channels, height, width)
    batch_size = 1
    height, width = 10, 10
    input_value = torch.randn(batch_size, test_config["in_channels"], height, width)
    
    # 7. 运行演示
    output = conv_demo(input_value, test_config)
    
    # 8. 打印结果形状
    logger.info(f"Input shape: {input_value.shape}")
    logger.info(f"Output shape: {output.shape}")
