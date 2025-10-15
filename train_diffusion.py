import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
    
    cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size // dist.get_world_size()
    if cfg.ddp.global_rank == 0:
        print_config(cfg)

    # ✅ Setup config
    cfg.diffusion.model.checkpoints_prefix = cfg.diffusion.model.checkpoints_prefix + \
        f"-{cfg.dataset.name}" \
        f"-{cfg.encoder.latent.num_latents}" \
        f"-d={cfg.diffusion.dynamic.d}" \
        f"-{cfg.suffix}"
    
    # ✅ Initialize Weights and Biases
    if not cfg.ddp.enabled or dist.get_rank() == 0:
        name = cfg.diffusion.model.checkpoints_prefix
        wandb.init(
            project=cfg.project.name,
            name=name,
            mode="online"
        )

    # ✅ Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = DiffusionTrainer(cfg)
    trainer.train()

    # ✅ Destroy DDP process group
    if cfg.ddp.enabled:
        time.sleep(300)
        dist.barrier()
        dist.destroy_process_group() 


if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()

"""
HYDRA_FULL_ERROR=1 \
PROJECT_ROOT=/home/vime725h/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12345 train_diffusion.py \
training=diffusion \
dataset=wikipedia-emnlp \
diffusion.model.num_workers=20 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=128-wikipedia-final-128/100000.pth"' 
"""