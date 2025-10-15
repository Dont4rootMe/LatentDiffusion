import os
import sys
import torch
import argparse
import torch.distributed as dist
import wandb
from datetime import timedelta

from encoder_trainer import EncoderTrainer
from utils import set_seed
from encoder_config import create_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoder arguments")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--latent_resolutions", type=str)
    args = parser.parse_args()
    
    config = create_config(args)
    config.ddp = False
    config.local_rank = 0
    config.optim.batch_size_per_gpu = config.optim.batch_size
    config.model.load_checkpoint = "./checkpoints/encoder-latent_dim=16-hidden_size=768-latent_resolutions=[8]/50000.pth"
    
    output_file = "text_reconstruction/result-[8].json"

    set_seed(config.seed)
    # wandb.login(key="11ea5000e27b950bf1e973cf9174baeda996f822")

    trainer = EncoderTrainer(config)
    trainer.reconstruction(args.latent_resolutions, output_file)

# python reconstruct.py --hidden_size=768 --latent_dim=16 --latent_resolutions="[1]"