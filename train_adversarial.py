import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time
from typing import Tuple, Dict
import torch

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, setup_ddp, print_config
from encoder_trainer import cross_entropy_loss, accuracy


class AdversarialTrainer(DiffusionTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.autoencoder.decoder.eval().cuda()

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        max_delta = self.cfg.diffusion.diffusion.T

        # Get latent
        batch = batch.to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            encoder_latents, _ = self.autoencoder.get_latent(batch, bert_output_masking=False)
            clean_x = self.autoencoder.normalize_latent(encoder_latents)

        # Add noise to the clean latent
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            adversarial_x = self.ddp_score_estimator(
                x_t=clean_x.clone(), 
                time_t=t.clone(),
                x_0_self_cond=torch.zeros_like(clean_x)
            )
        
        batch_size = clean_x.size(0)
        dim = (clean_x.size(1) * clean_x.size(2)) ** 0.5
        r = torch.min(torch.ones(batch_size, device=clean_x.device), 
                      t / torch.norm(clean_x - adversarial_x, dim=(1, 2)) * dim)
        adversarial_x = clean_x + r[:, None, None] * (adversarial_x - clean_x)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            adversarial_x = self.autoencoder.denormalize_latent(adversarial_x)
            logits, _ = self.autoencoder.decoder(
                encoder_latents=adversarial_x, 
                return_last_hidden_state=True,
            ) 

        loss = -cross_entropy_loss(
            input=logits,
            target=batch["input_ids"],
            mask=batch["attention_mask"],
        )

        acc = accuracy(
                logits=logits,
                target=batch["input_ids"],
                mask=batch["attention_mask"]
        )

        self.log_metric("statistics", "accuracy", acc.detach().item())
        self.log_metric("statistics", "loss", loss.detach().item())

        return loss, {}




@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
    
    cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size // dist.get_world_size()
    if cfg.ddp.global_rank == 0:
        print_config(cfg)

    # ✅ Setup config
    cfg.diffusion.model.checkpoints_prefix = "adversarial" \
        f"-{cfg.dataset.name}" \
        f"-delta={cfg.diffusion.diffusion.T}" \
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
    trainer = AdversarialTrainer(cfg)
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
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=8 --master_port=12345 train_adversarial.py \
project.name=latent-diffusion-adversarial-v1.0 \
training=diffusion \
dataset=wikipedia \
encoder.latent.num_latents=16 \
decoder.latent.num_latents=16 \
diffusion.diffusion.T=0.3 \
diffusion.training.training_iters=10000 \
diffusion.training.batch_size=1024 \
diffusion.logging.save_freq=10000 \
diffusion.logging.eval_freq=100000 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-v1.0-128/100000.pth"' \
suffix=v1.0
"""