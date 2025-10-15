import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time

from decoder_trainer import DecoderTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
    
    cfg.autoencoder.training.batch_size_per_gpu = cfg.autoencoder.training.batch_size // dist.get_world_size()
    if cfg.ddp.global_rank == 0:
        print_config(cfg)

    # ✅ Setup config
    cfg.autoencoder.model.checkpoints_prefix = cfg.autoencoder.model.load_checkpoint.split("/")[-2] + f"-finetuned-{cfg.suffix}"
    
    # ✅ Initialize Weights and Biases
    if not cfg.ddp.enabled or dist.get_rank() == 0:
        name = cfg.autoencoder.model.checkpoints_prefix
        wandb.init(
            project=cfg.project.name,
            name=name,
            mode="online"
        )

    # ✅ Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = DecoderTrainer(cfg)
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
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=2 --master_port=12346 finetune_decoder.py \
dataset=wikipedia \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-v2.3.4-128/100000.pth"' \
decoder.finetuning.max_std=1.0 \
decoder.finetuning.is_alpha=true \
decoder.finetuning.latent_masking=false \
decoder.finetuning.bert_output_masking=false \
suffix="v2.5.3.10"
"""