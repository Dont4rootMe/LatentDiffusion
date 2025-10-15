import torch
from diffusion_trainer import DiffusionTrainer
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import sys
import pickle
from hydra.core.global_hydra import GlobalHydra
from evaluate import load
import numpy as np

from utils import seed_everything


#########################################################################################################
##########################       Interpolation between two latent vectors      ##########################
#########################################################################################################

def run_experiment(trainer: DiffusionTrainer):
    trainer.autoencoder.decoder.cuda()
    steps_range = list(range(1, 200, 1))

    results = {
        "steps": [],
        "pred_text": [],
    }
    batch_size = 100

    for steps in steps_range:
        trainer.cfg.diffusion.dynamic.N = steps
        pred_text, _ = trainer.generate_text_batch(batch_size=batch_size)
        results["steps"].append(steps)
        results["pred_text"].append(pred_text)
    
    df = pd.DataFrame(results)
    df.to_csv(f"experiments/ddim_generation-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv", index=False)


##############################################################################
##########################       Setup Config       ##########################
##############################################################################

def setup_config():
    PROJECT_ROOT = "/home/jovyan/vmeshchaninov/LatentDiffusion"
    sys.path.append(PROJECT_ROOT)
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT

    GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"../conf", version_base=None)  # Set path to your configs
    cfg = hydra.compose(config_name="config", overrides=["dataset=rocstories"])  # Replace with your main config file

    cfg.ddp.enabled = False

    # Parameters for integration
    cfg.encoder.latent.num_latents = 16
    cfg.decoder.latent.num_latents = 16
    cfg.diffusion.dynamic.d = 5
    # prefix_folder = "/home/jovyan/shares/SR004.nfs2/vmeshchaninov/HierarchicalDiffusion/checkpoints_v2"
    prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"

    cfg.diffusion.dynamic.solver = "ddim"
    cfg.diffusion.training.batch_size_per_gpu = 100

    # GOOD CHECKPOINTS
    diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.1/100000.pth"
    autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.1-128/100000.pth"
    
    # BAD CHECKPOINTS
    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.0/100000.pth"
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.0-128/100000.pth"

    cfg.diffusion.model.load_checkpoint = os.path.join(prefix_folder, diffusion_checkpoints)
    cfg.autoencoder.model.load_checkpoint = os.path.join(prefix_folder, autoencoder_checkpoints)
    cfg.training = ""

    return cfg


def main(cfg: DictConfig, args: dict = None):
    seed = cfg.project.seed
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = DiffusionTrainer(cfg)
    trainer.restore_checkpoint()

    run_experiment(trainer)


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=0 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12346 -m experiments.ddim_generation \
dataset=rocstories
"""