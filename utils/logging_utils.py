import wandb
import torch
import os
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List

def print_config(cfg: DictConfig):
    """Print Hydra configuration accurately."""
    
    print(OmegaConf.to_yaml(cfg, resolve=False)) 


def config_to_wandb(cfg: DictConfig):
    config_path = "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Log the config file as an artifact in W&B
    artifact = wandb.Artifact(name="hydra_config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)

    print("📂 Hydra config logged to W&B as an artifact!")


def log_batch_of_tensors_to_wandb(batch_of_tensors: Dict[str, torch.Tensor]):
    """Log a batch of tensors to W&B."""
    batch_index = 0
    columns = sorted(batch_of_tensors.keys())
    seq_len = batch_of_tensors[columns[0]].shape[1]
    data = [tuple(batch_of_tensors[col][batch_index][i].detach().cpu().item() for col in columns) for i in range(seq_len)]

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"token_table": table})


def log_batch_of_texts_to_wandb(batch_of_texts: List[str]):
    # Convert list of strings to list of tuples for wandb.Table
    table = wandb.Table(columns=["text"])
    for text in batch_of_texts:
        table.add_data(text)
    wandb.log({"generated_texts": table})


def init_json_logger(cfg: DictConfig, experiment_name: str = None):
    """
    Initialize JSON logger as a drop-in replacement for wandb.
    Saves logs in the same directory structure as checkpoints.
    """
    from . import wandb_json_logger
    
    # Create experiment name based on config
    if experiment_name is None:
        if hasattr(cfg, 'autoencoder') and hasattr(cfg.autoencoder.model, 'checkpoints_prefix'):
            experiment_name = cfg.autoencoder.model.checkpoints_prefix
        elif hasattr(cfg, 'diffusion') and hasattr(cfg.diffusion.model, 'checkpoints_prefix'):
            experiment_name = cfg.diffusion.model.checkpoints_prefix
        else:
            experiment_name = "experiment"
    
    # Use checkpoint directory as base for logs
    checkpoint_dir = cfg.project.checkpoint_dir
    log_dir = os.path.join(checkpoint_dir, "logs")
    
    # Initialize the logger
    logger = wandb_json_logger.init(
        project=cfg.project.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=log_dir,
        experiment_name=experiment_name
    )
    
    print(f"📂 JSON Logger initialized. Logs will be saved to: {log_dir}")
    return logger


def config_to_json_logger(cfg: DictConfig):
    """
    Log config to JSON logger (equivalent to config_to_wandb).
    """
    from . import wandb_json_logger
    
    # Convert config to dict and set it
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_json_logger.set_config(config_dict)
    
    print("📂 Hydra config logged to JSON logger!")