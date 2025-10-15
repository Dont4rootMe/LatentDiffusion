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


##################################################################################################################
##########################       Reconstruction Loss between predicted x_0 and x_0      ##########################
##################################################################################################################

@torch.no_grad()
def run_experiment(trainer: DiffusionTrainer):
    trainer._setup_valid_data_generator()

    batch = next(iter(trainer.valid_loader))
    batch = batch.to(trainer.device)

    with torch.no_grad():
        latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.autoencoder.normalize_latent(latent)

    print(f"Std of clean_x: {torch.std(clean_x, dim=0).mean()}")

    # Create dict to store losses for each t
    results = {
        "t": [],
        "loss_x_0": []
    }

    # Iterate through t values from 0 to 1
    for t_val in tqdm(torch.linspace(0, 1, 100)):
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
        results["loss_x_0"].append(loss_x_0.item())
        results["t"].append(t_val.item())

    df = pd.DataFrame(results)
    df.to_csv(f"experiments/reconstruction_loss-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv", index=False)
    

#########################################################################################################
##########################       Bert Score between predicted x_0 and x_0      ##########################
#########################################################################################################
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

class SentenceMetric:
    def __init__(self,):
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.model.cuda()

    def __call__(self, text1, text2):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            output1 = self.model.encode(text1, normalize_embeddings=True)
            output2 = self.model.encode(text2, normalize_embeddings=True)
        
            return torch.sum(output1 * output2, dim=1).mean()



def get_text(trainer, pred_embeddings):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_latents = trainer.autoencoder.denormalize_latent(pred_embeddings)
        pred_logits = trainer.autoencoder.decoder(pred_latents)

    text, _ = trainer.sample_from_logits(pred_logits)

    return text


@torch.no_grad()
def run_experiment_2(trainer: DiffusionTrainer):
    trainer.cfg.diffusion.training.batch_size = 100
    trainer._setup_valid_data_generator()

    trainer.autoencoder.decoder.cuda()

    # bertscore = load("bertscore", module_type="metric")

    # sentence_metric = SentenceMetric()

    batch = next(iter(trainer.valid_loader))
    batch = batch.to(trainer.device)

    trg_text = batch["text_trg"]

    with torch.no_grad():
        latent, bert_hidden_state = trainer.autoencoder.get_latent(batch, bert_output_masking=False)
        clean_x = trainer.autoencoder.normalize_latent(latent)

    print(f"Std of clean_x: {torch.std(clean_x, dim=0).mean()}")

    # Create dict to store losses for each t
    results = {
        "t": [],
        "bert-score": [],
        "pred_text": [],
        "trg_text": [],
    }

    batch_size = clean_x.shape[0]

    # Iterate through t values from 0 to 1
    for t_val in tqdm(torch.linspace(0, 1, 100)):
        t = torch.full((batch_size,), t_val).cuda()
        
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
        pred_text = get_text(trainer, x_0)
        results["pred_text"].append(pred_text)
        results["trg_text"].append(trg_text)
        results["t"].append(t_val.item())
    

    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     output = bertscore.compute(predictions=all_pred_text, references=all_trg_text, lang='en', verbose=True)

    # bert_scores = []
    # for i in range(0, len(all_pred_text), batch_size):
    #     pred_text = all_pred_text[i:i+batch_size]
    #     trg_text = all_trg_text[i:i+batch_size]
    #     bert_scores.append(sentence_metric(pred_text, trg_text))

    df = pd.DataFrame(
        {
            "t": results["t"],
            "pred_text": results["pred_text"],
            "trg_text": results["trg_text"],
        }
    )
    df.to_csv(f"experiments/bert_score-{trainer.cfg.diffusion.model.load_checkpoint.split('/')[-2]}.csv", index=False)
    




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

    run_experiment_2(trainer)


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)


"""
CUDA_VISIBLE_DEVICES=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=1 --master_port=12345 -m experiments.reconstruction_loss \
dataset=rocstories
"""