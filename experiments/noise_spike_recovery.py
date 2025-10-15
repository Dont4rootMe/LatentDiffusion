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
from argparse import ArgumentParser
from utils import seed_everything


#########################################################################################################
##########################       Interpolation between two latent vectors      ##########################
#########################################################################################################

def run_generation(trainer: DiffusionTrainer, batch_size: int, spike_t: float, spike_std: float, noise):
    trainer.score_estimator.eval()

    num_latents = trainer.cfg.encoder.latent.num_latents
    shape = (
        batch_size,
        num_latents,
        trainer.cfg.encoder.latent.dim
    )

    seed_everything(42)

    x_t = trainer.dynamic.prior_sampling(shape).to(trainer.device)
    x_0_self_cond = torch.zeros_like(x_t)

    with trainer.ema.average_parameters(), torch.no_grad():
        timesteps = torch.linspace(
            trainer.cfg.diffusion.diffusion.T, 
            trainer.cfg.diffusion.diffusion.t_min, 
            trainer.cfg.diffusion.dynamic.N + 1, 
            device=trainer.device
        )

        trajectory = []
        is_spike = False
        
        for idx in tqdm(range(trainer.cfg.diffusion.dynamic.N)):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]

            if t >= spike_t and next_t <= spike_t and not is_spike:
                is_spike = True
                x_t = x_t + noise * spike_std

            input_t = t * torch.ones(shape[0], device=trainer.device)
            next_input_t = next_t * torch.ones(shape[0], device=trainer.device)

            output = trainer.diff_eq_solver.step(
                x_t=x_t, t=input_t, next_t=next_input_t,
                x_0_self_cond=x_0_self_cond,
            )

            x_t, x_mean = output["x"], output["x_mean"]
            x_0_self_cond = output["x_0"]

            trajectory.append(x_0_self_cond.detach())

    return trajectory

def run_experiment(trainer: DiffusionTrainer):
    t_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    std_range = [0.05, 0.1, 0.2, 0.5]

    df = pd.DataFrame(index=std_range, columns=t_range, dtype=float)

    batch_size = 100

    noise = torch.randn(batch_size, 16, 768, device=trainer.device)

    for t in t_range:
        real_trajectory = run_generation(trainer, batch_size, t, 0, noise)

        for std in std_range:
            noised_trajectory = run_generation(trainer, batch_size, t, std, noise)

            mse = (real_trajectory[-1] - noised_trajectory[-1]).square().mean().item()
            #[(t1 - t2).square().mean().item() for t1, t2 in zip(real_trajectory, noised_trajectory)]
            df.loc[std, t] = mse
            
    df.to_csv(f"experiments/noise_spike_recovery-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv")


##############################################################################
##########################       Setup Config       ##########################
##############################################################################

def setup_config(args: dict):
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

    cfg.diffusion.dynamic.N = 100
    cfg.diffusion.diffusion.t_min = 0.05
    cfg.diffusion.dynamic.solver = "ddim"

    # GOOD CHECKPOINTS
    if args.suffix == "v2.3.4":
        diff_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
        autoencoder_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
        diffusion_checkpoints = "diffusion-rocstories-16-d=5-v2.3.4/180000.pth"
        autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.3.4-128/100000.pth"
    elif args.suffix == "v1.0":
        diff_prefix_folder = "/home/jovyan/shares/SR004.nfs2/vmeshchaninov/vmeshchaninov/LatentDiffusion/checkpoints_04_26"
        autoencoder_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
        diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.0/180000.pth"
        autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.0-128/100000.pth"
    
    # BAD CHECKPOINTS
    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.0/100000.pth"
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.0-128/100000.pth"

    cfg.diffusion.model.load_checkpoint = os.path.join(diff_prefix_folder, diffusion_checkpoints)
    cfg.autoencoder.model.load_checkpoint = os.path.join(autoencoder_prefix_folder, autoencoder_checkpoints)
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
    argparse = ArgumentParser()
    argparse.add_argument("--suffix", type=str, default="v1.1")
    args = argparse.parse_args()

    cfg = setup_config(args)
    cfg.suffix = args.suffix
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12345 -m experiments.noise_spike_recovery \
--suffix v2.3.4
"""