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

def get_text(trainer, pred_embeddings):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_latents = trainer.autoencoder.denormalize_latent(pred_embeddings)
        pred_logits = trainer.autoencoder.decoder(pred_latents)

    text, _ = trainer.sample_from_logits(pred_logits)

    return text


def get_batch_of_latents(trainer: DiffusionTrainer):
    for batch in trainer.valid_loader:
        batch = batch.to(trainer.device)
        with torch.no_grad():
            latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
            clean_x = trainer.autoencoder.normalize_latent(latent)
        yield clean_x


@torch.no_grad()
def run_experiment(trainer: DiffusionTrainer):
    trainer.autoencoder.decoder.cuda()
    trainer._setup_valid_data_generator()

    iter_latents = get_batch_of_latents(trainer)

    latent_1 = next(iter_latents)
    latent_2 = next(iter_latents)

    assert not torch.allclose(latent_1, latent_2)

    results = {}

    batch_size = latent_1.shape[0]

    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Create dict to store losses for each t
        time_results = {
            "alpha": [],
            "pred_text": [],
            "x_t_texts": [],
        }

        input_t = torch.full((batch_size,), t).cuda()
        n_points = 100
        for alpha in tqdm(torch.linspace(0, 1, n_points)):
            latent_m = latent_1 * alpha + latent_2 * (1 - alpha)
            seed_everything(42)
            marg_forward = trainer.dynamic.marginal(latent_m, input_t)
            x_t, noise = marg_forward['x_t'], marg_forward['noise']

            # self-cond estimate
            x_0_self_cond = torch.zeros_like(latent_m, dtype=latent_m.dtype)
            if trainer.cfg.diffusion.diffusion.use_self_cond:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                    x_0_self_cond = trainer.score_estimator(
                        x_t=x_t.clone(),
                        time_t=input_t.clone(), 
                        x_0_self_cond=x_0_self_cond
                    ).detach()

            # model prediction
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                x_0 = trainer.score_estimator(
                    x_t=x_t,
                    time_t=input_t,
                    x_0_self_cond=x_0_self_cond
                )

            # Calculate and store MSE loss for this t
            pred_text = get_text(trainer, x_0)
            time_results["pred_text"].append(pred_text)
            time_results["x_t_texts"].append(get_text(trainer, x_t))
            time_results["alpha"].append(alpha.item())
        
        results[t] = time_results

    # df = pd.DataFrame(results)
    import json
    json.dump(results, open(f"experiments/interpolation-all_times-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.json", "w"))
    # df.to_csv(f"experiments/interpolation-all_times-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv", index=False)


def generation_from_latent(trainer: DiffusionTrainer, x_t: torch.Tensor, t_start: float, t_end: float, n_steps: int):
    trainer.score_estimator.eval()

    shape = x_t.shape
    seed_everything(42)
    x_0_self_cond = torch.zeros_like(x_t)

    with trainer.ema.average_parameters(), torch.no_grad():
        timesteps = torch.linspace(
            t_start, 
            t_end, 
            n_steps + 1, 
            device=trainer.device
        )

        for idx in range(n_steps):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]

            input_t = t * torch.ones(shape[0], device=trainer.device)
            next_input_t = next_t * torch.ones(shape[0], device=trainer.device)

            output = trainer.diff_eq_solver.step(
                x_t=x_t, t=input_t, next_t=next_input_t,
                x_0_self_cond=x_0_self_cond,
            )

            x_t, x_mean = output["x"], output["x_mean"]
            x_0_self_cond = output["x_0"]

    return x_0_self_cond


@torch.no_grad()
def run_experiment_2(trainer: DiffusionTrainer):
    trainer.autoencoder.decoder.cuda()
    trainer._setup_valid_data_generator()

    batch = next(iter(trainer.valid_loader))
    batch = batch.to(trainer.device)

    with torch.no_grad():
        latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.autoencoder.normalize_latent(latent)
    
    batch_size = 100

    latent_1 = clean_x[:batch_size]
    latent_2 = clean_x[batch_size:2*batch_size]

    # Create dict to store losses for each t
    results = {
        "alpha": [],
        "pred_text": [],
    }

    # Iterate through t values from 0 to 1
    t = 0.1
    input_t = torch.full((batch_size,), t).cuda()
    n_points = 100
    for alpha in tqdm(torch.linspace(0, 1, n_points)):
        latent_m = latent_1 * alpha + latent_2 * (1 - alpha)
        seed_everything(42)
        marg_forward = trainer.dynamic.marginal(latent_m, input_t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        x_0 = generation_from_latent(trainer, x_t, t, 0.05, 10)

        # Calculate and store MSE loss for this t
        pred_text = get_text(trainer, x_0)
        results["pred_text"].append(pred_text)
        results["alpha"].append(alpha.item())

    df = pd.DataFrame(results)
    df.to_csv(f"experiments/interpolation-generation-from-latent-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}-t={t}.csv", index=False)


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
    diff_prefix_folder = "/home/jovyan/shares/SR004.nfs2/vmeshchaninov/vmeshchaninov/LatentDiffusion/checkpoints_04_26"
    autoencoder_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
    # prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"

    # GOOD CHECKPOINTS
    diff_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
    diffusion_checkpoints = "diffusion-rocstories-16-d=5-v2.3.4/180000.pth"
    autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.3.4-128/100000.pth"
    
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
    cfg = setup_config()
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12345 -m experiments.interpolation \
dataset=rocstories
"""