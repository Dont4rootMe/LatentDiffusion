import torch
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
from torch.nn.functional import cross_entropy
from evaluate import load

from utils import seed_everything
from encoder_trainer import accuracy
from encoder_trainer import EncoderTrainer


def sample_from_logits(trainer, logits):
    eos_token_id = trainer.tokenizer.eos_token_id 
    if eos_token_id is None:
        eos_token_id = trainer.tokenizer.sep_token_id
    
    tokens = logits.argmax(dim=-1).detach().cpu().tolist()
                
    tokens_list = []
    for seq_list in tokens:
        for ind, token in enumerate(seq_list):
            if token == eos_token_id:
                tokens_list.append(seq_list[:ind])
                break
        else:
            tokens_list.append(seq_list)

    return trainer.tokenizer.batch_decode(tokens_list, skip_special_tokens=True), tokens


@torch.no_grad()
def get_text(trainer, pred_embeddings):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_latents = trainer.denormalize_latent(pred_embeddings)
        pred_logits = trainer.decoder(pred_latents)

    text, _ = sample_from_logits(trainer, pred_logits)

    return text


@torch.no_grad()
def run_experiment(trainer: EncoderTrainer):
    bleu = load("bleu")


    # Load model
    trainer.encoder.cuda()
    trainer.encoder.eval()
    trainer.decoder.cuda()
    trainer.decoder.eval()
    trainer._setup_valid_data_generator()

    batch = next(iter(trainer.valid_loader))
    batch_size = 100
    batch = batch.to(trainer.device)

    batch["input_ids"] = batch["input_ids"][:batch_size]
    batch["attention_mask"] = batch["attention_mask"][:batch_size]

    with torch.no_grad():
        latent, _ = trainer.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.normalize_latent(latent)

    references = get_text(trainer, clean_x)

    # Create dict to store losses for each t
    results = {
        "delta": [],
        "loss": [],
        "acc": [],
        "bleu": [],
    }

    # Iterate through t values from 0 to 1
    max_delta = 4.0
    n_points = 100
    for t in tqdm(torch.linspace(0, max_delta, n_points)):
        seed_everything(42)

        adversarial_x = clean_x + t * torch.randn_like(clean_x)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            denorm_adversarial_x = trainer.denormalize_latent(adversarial_x)
            logits, _ = trainer.decoder(
                encoder_latents=denorm_adversarial_x, 
                return_last_hidden_state=True,
            ) 

        seq_len = batch["input_ids"].size(1)
        max_loss = 10

        losses = cross_entropy(
            input=logits[:, :seq_len].reshape(-1, logits.shape[-1]), 
            target=batch["input_ids"][:, :seq_len].reshape(-1), 
            reduction="none"
        )
        losses = losses * batch["attention_mask"][:, :seq_len].reshape(-1)
        losses = torch.min(losses, torch.tensor(max_loss, device=losses.device))
        loss = torch.mean(losses) * (-1)

        acc = accuracy(
                logits=logits[:, :seq_len],
                target=batch["input_ids"][:, :seq_len],
                mask=batch["attention_mask"][:, :seq_len]
        )
        predictions = get_text(trainer, adversarial_x)
        bleu_results = bleu.compute(predictions=predictions, references=references)

        results["delta"].append(t.item())
        results["loss"].append(loss.item())
        results["acc"].append(acc.item())
        results["bleu"].append(bleu_results["bleu"])

    df = pd.DataFrame(results)
    df.to_csv(f"experiments/decoder-robustness-{trainer.cfg.autoencoder.model.load_checkpoint.split('/')[-2]}.csv", index=False)




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


    autoencoder_prefix_folder = "/home/jovyan/vmeshchaninov/LatentDiffusion/checkpoints"

    # GOOD CHECKPOINTS
    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.0-128/100000.pth"


    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.2.7-128/100000.pth"


    # autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v2.3.4-128/100000.pth"

    
    # BAD CHECKPOINTS
    autoencoder_checkpoints = "autoencoder-num_latents=16-wikipedia-v1.0-128/20000.pth"

    cfg.autoencoder.model.load_checkpoint = os.path.join(autoencoder_prefix_folder, autoencoder_checkpoints)
    cfg.training = ""

    return cfg


def main(cfg: DictConfig, args: dict = None):
    seed = cfg.project.seed
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = EncoderTrainer(cfg)
    trainer.restore_checkpoint()

    run_experiment(trainer)


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12345 -m experiments.adversarial_perturbations \
dataset=wikipedia
"""