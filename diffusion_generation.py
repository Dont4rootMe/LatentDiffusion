import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time
import json
import os
import torch

from diffusion_trainer import DiffusionTrainer, gather_texts
from utils import seed_everything, setup_ddp, print_config


import os
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"


@torch.no_grad()
def estimate(self,):
    # Generation
    print("Generating texts...")
    if not self.cfg.ddp.enabled:
        num_texts = self.cfg.diffusion.generation.num_gen_texts
    else:
        num_texts = self.cfg.diffusion.generation.num_gen_texts // dist.get_world_size()
        if dist.get_rank() < self.cfg.diffusion.generation.num_gen_texts % dist.get_world_size():
            num_texts += 1
    
    result_dict = self.generate_text(num_texts=num_texts)

    # Gathering
    if self.cfg.ddp.enabled:
        for key in result_dict:
            result_dict[key] = gather_texts(result_dict[key])
            result_dict[key] = result_dict[key][:self.cfg.diffusion.generation.num_gen_texts]

    # Logging
    list_of_dicts = [{key: result_dict[key][i] for key in ["TRG", "GEN"]} for i in range(len(result_dict["TRG"]))]
            
    if not self.cfg.ddp.enabled or dist.get_rank() == 0:
        dir_path = os.path.join(self.cfg.project.path, self.cfg.diffusion.generation.texts_dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        prefix_folder = os.path.join(dir_path, self.cfg.autoencoder.model.load_checkpoint.split("/")[0])
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)
        
        total_len = len(list_of_dicts)
        file_name = f"N={self.cfg.diffusion.dynamic.N}-len={total_len}.json"
        save_path = os.path.join(prefix_folder, file_name)
        json.dump(list_of_dicts, open(save_path, "w"), indent=4)
        print(f"Texts are saved in {save_path}")

    # Metrics
    mauve_value = self._compute_mauve(result_dict["GEN"], result_dict["TRG"])
    ppl_value = self._compute_ppl(result_dict["GEN"])
    div_value = self._compute_div(result_dict["GEN"])

    if dist.get_rank() == 0:
        metrics_path = save_path.replace(".json", "_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Mauve: {mauve_value:0.5f}\n")
            f.write(f"PPL: {ppl_value:0.5f}\n")
            f.write(f"DIV: {div_value:0.5f}\n")
        print(f"Metrics are saved in {metrics_path}")
    torch.cuda.synchronize()


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

    # ✅ Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = DiffusionTrainer(cfg)
    trainer.restore_checkpoint()

    # ✅ Estimate
    estimate(trainer)

if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()

"""
HYDRA_FULL_ERROR=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=2 --master_port=12345 diffusion_generation.py \
training=None \
dataset=rocstories \
encoder.latent.num_latents=32 \
decoder.latent.num_latents=32 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=32-wikipedia-final-128/100000.pth"' \
diffusion.model.load_checkpoint='"diffusion-rocstories-32-d=5-final/120000.pth"' \
diffusion.generation.num_gen_texts=1000 \
suffix=final
"""