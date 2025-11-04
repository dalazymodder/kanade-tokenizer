import torch
from lightning.pytorch.cli import LightningCLI

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    if torch.__version__ >= "2.0":
        torch._dynamo.config.cache_size_limit = 1000

    cli = LightningCLI(save_config_kwargs={"overwrite": True})
