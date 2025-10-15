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
from utils import seed_everything
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def get_latents(trainer: DiffusionTrainer, texts: list[str]):
    batch_size = 1000

    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=trainer.autoencoder.collate_fn)
    latents = []
    for batch in tqdm(loader):
        batch = batch.to(trainer.device)
        with torch.no_grad():
            encoder_latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
            clean_x = trainer.autoencoder.normalize_latent(encoder_latent)
            latents.extend(clean_x.detach().cpu().tolist())
    return latents

@torch.no_grad()
def run_experiment(trainer: DiffusionTrainer):
    dataset = load_from_disk(trainer.cfg.dataset.dataset_path)

    train_latents = get_latents(trainer, dataset["train"])
    test_latents = get_latents(trainer, dataset["test"])

    dt = DatasetDict({
        "train": Dataset.from_list([{"latent": latent, "text": text} for latent, text in zip(train_latents, dataset["train"]["text_trg"])]),
        "test": Dataset.from_list([{"latent": latent, "text": text} for latent, text in zip(test_latents, dataset["test"]["text_trg"])])
    })
    dt.save_to_disk(f"./experiments/{trainer.cfg.dataset.name}-latents-{trainer.cfg.autoencoder.model.load_checkpoint.split('/')[-2]}")
    
    

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
torchrun --nproc_per_node=1 --master_port=12345 -m experiments.get_latents \
dataset=rocstories
"""