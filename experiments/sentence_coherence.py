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
import nltk
from copy import deepcopy

from utils import seed_everything


def get_cummulative_loss(trainer: DiffusionTrainer, batch: torch.Tensor):
    batch = batch.to(trainer.device)

    with torch.no_grad():
        latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.autoencoder.normalize_latent(latent)

    print(f"Std of clean_x: {torch.std(clean_x, dim=0).mean()}")

    cummulative_loss = 0
    num_steps = 100

    # Iterate through t values from 0 to 1
    for t_val in tqdm(torch.linspace(0, 1, num_steps)):
        t = torch.full((clean_x.shape[0],), t_val).cuda()
        
        marg_forward = trainer.dynamic.marginal(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if trainer.cfg.diffusion.diffusion.use_self_cond:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                x_0_self_cond = trainer.score_estimator(
                    x_t=x_t.clone(),
                    time_t=t.clone(), 
                    x_0_self_cond=x_0_self_cond
                ).detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            x_0 = trainer.score_estimator(
                x_t=x_t,
                time_t=t,
                x_0_self_cond=x_0_self_cond
            )

        # Calculate and store MSE loss for this t
        loss_x_0 = torch.mean(torch.square(clean_x - x_0))
        cummulative_loss += loss_x_0
    
    return cummulative_loss / num_steps



########################################################################################################
##########################       Corruption using shuffling of sentences      ##########################
########################################################################################################

def shuffle_sentences(texts: list[str], n: int):
    shuffled_texts = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)

        order = list(range(len(sentences)))
        permutation = np.random.choice(order, size=min(n, len(sentences)), replace=False)
        sorted_permutation = np.sort(permutation)

        # 1, 2, 3
        # 3, 1, 2
        

        shuffled_sentences = deepcopy(sentences)
        for i, j in zip(sorted_permutation, permutation):
            shuffled_sentences[i] = sentences[j]

        shuffled_text = " ".join(shuffled_sentences)
        shuffled_texts.append(shuffled_text)
    
    return shuffled_texts

@torch.no_grad()
def run_experiment(trainer: DiffusionTrainer):
    trainer._setup_valid_data_generator()

    batch = next(iter(trainer.valid_loader))
    texts = batch["text_trg"]

    results = {
        "n": [],
        "loss": []
    }

    for n in range(0, 5):
        shuffled_texts = shuffle_sentences(texts, n)

        batch = trainer.tokenizer(
            shuffled_texts,
            add_special_tokens=trainer.cfg.tokenizer.add_special_tokens,
            padding=trainer.cfg.tokenizer.padding,
            truncation=trainer.cfg.tokenizer.truncation,
            max_length=trainer.cfg.dataset.max_sequence_len,
            return_tensors=trainer.cfg.tokenizer.return_tensors,
            return_attention_mask=trainer.cfg.tokenizer.return_attention_mask,
            return_token_type_ids=trainer.cfg.tokenizer.return_token_type_ids,
        )

        results["n"].append(n)
        results["loss"].append(get_cummulative_loss(trainer, batch).item())


    df = pd.DataFrame(results)
    df.to_csv(f"experiments/sentence_coherence-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv", index=False)

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
    # diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.1/100000.pth"
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.1-128/100000.pth"
    
    # BAD CHECKPOINTS
    diffusion_checkpoints = "diffusion-rocstories-16-d=5-v1.0/100000.pth"
    autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.0-128/100000.pth"

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
torchrun --nproc_per_node=1 --master_port=12343 -m experiments.sentence_coherence \
dataset=rocstories
"""