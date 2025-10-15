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


def get_latent_from_texts(trainer: DiffusionTrainer, texts: list[str]):
    dict_texts = [{"text_trg": text} for text in texts]
    batch = trainer.autoencoder.collate_fn(dict_texts)
    batch = batch.to(trainer.device)
    mask = batch["attention_mask"]
    with torch.no_grad():
        encoder_latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.autoencoder.normalize_latent(encoder_latent)
    return clean_x, mask


def run_experiment(trainer: DiffusionTrainer):
    trainer.autoencoder.decoder.cuda()
    
    batch_size = 100

    results = {}

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
        pred_text, pred_embeddings, pred_logits = trainer.generate_text_batch(batch_size=batch_size)
        training_pred_embeddings, training_mask = get_latent_from_texts(trainer, pred_text)

    # compute MSE
    mse = (training_pred_embeddings - pred_embeddings).square().mean()
    results["mse"] = mse.item()

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        # compute entropy directly from logits using the log-sum-exp trick (no explicit softmax)
        log_z = torch.logsumexp(pred_logits, dim=-1)                              # [batch, seq_len]
        log_z[log_z.isinf()] = torch.finfo(torch.float32).max

        numerator = torch.sum(pred_logits * torch.exp(pred_logits), dim=-1)       # [batch, seq_len]
        numerator[numerator.isinf()] = torch.finfo(torch.float32).max

        weighted_sum = numerator / torch.exp(log_z) 

        entropy = (log_z - weighted_sum)
        seq_len = training_mask.shape[1]
        entropy = (entropy[:, :seq_len] * training_mask).sum() / training_mask.sum()
        results["entropy"] = entropy.item()
    # probs = torch.softmax(pred_logits, dim=-1)
    # entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
    # results["entropy"] = entropy.item()

    print(f"MSE: {results['mse']:.4f}, Entropy: {results['entropy']:.4f}")
    
    

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
    cfg.diffusion.diffusion.t_min = 0.05
    cfg.diffusion.dynamic.N = 200

    autoencoder_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"
    diff_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"


    # GOOD CHECKPOINTS
    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v2.0/180000.pth"
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.0-128/100000.pth"


    diffusion_checkpoints = "diffusion-rocstories-16-d=5-v2.2.7/180000.pth"
    autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.2.7-128/100000.pth"


    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v2.3.4/180000.pth"
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.3.4-128/100000.pth"

    
    # BAD CHECKPOINTS
    # diff_prefix_folder = "/home/jovyan/shares/SR004.nfs2/vmeshchaninov/vmeshchaninov/LatentDiffusion/checkpoints_04_26"
    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.0/200000.pth"
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
CUDA_VISIBLE_DEVICES=0 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12346 -m experiments.train_test_mismatch \
dataset=rocstories
"""