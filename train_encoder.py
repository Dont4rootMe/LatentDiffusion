import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time
from encoder_trainer import EncoderTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    # if cfg.ddp.enabled:
    #     cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
    
    # cfg.autoencoder.training.batch_size_per_gpu = cfg.autoencoder.training.batch_size // dist.get_world_size()
    # if cfg.ddp.global_rank == 0:
    #     print_config(cfg)

    # ✅ Setup config
    # cfg.autoencoder.model.checkpoints_prefix = cfg.autoencoder.model.checkpoints_prefix + \
    #     f"-num_latents={cfg.encoder.latent.num_latents}" \
    #     f"-{cfg.dataset.name}" \
    #     f"-{cfg.suffix}-{cfg.dataset.max_sequence_len}"

    # ✅ Initialize Weights and Biases
    # if not cfg.ddp.enabled or dist.get_rank() == 0:
    #     name = cfg.autoencoder.model.checkpoints_prefix
    #     wandb.init(
    #         project=cfg.project.name,
    #         name=name,
    #         mode="online"
    #     )

    # ✅ Seed everything
    seed = cfg.project.seed # + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = EncoderTrainer(cfg)
    trainer.train()

    # ✅ Destroy DDP process group
    # if cfg.ddp.enabled:
    #     time.sleep(300)
    #     dist.barrier()
    #     dist.destroy_process_group() 


if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()

"""
# reconstruction training on wikipedia-512
HYDRA_FULL_ERROR=1 \
PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=2 --master_port=12346 train_encoder.py \
dataset=openwebtext-512 \
encoder.latent.num_latents=32 \
encoder.embedding.max_position_embeddings=512 \
decoder.latent.num_latents=32 \
decoder.embedding.max_position_embeddings=512 \
autoencoder.training.training_iters=200000 \
suffix="final"
"""


"""
# reconstruction training on wikipedia

PROJECT_ROOT=/home/jovyan/vmeshchaninov/LatentDiffusion \
torchrun --nproc_per_node=2 --master_port=12346 train_encoder.py \
dataset=wikipedia \
encoder.latent.num_latents=16 \
decoder.latent.num_latents=16 \
encoder.augmentation.masking.weight=0.5 \
encoder.augmentation.masking.encodings_mlm_probability=0.3 \
encoder.augmentation.gaussian_noise.weight=0.5 \
encoder.augmentation.gaussian_noise.delta=0.5 \
suffix="v2.2.5"
"""


"""
PROJECT_ROOT=/home/jovyan/vmeshchaninov/HierarchicalDiffusion \
torchrun --nproc_per_node=2 --master_port=12346 train_encoder.py \
dataset=wikipedia \
dataset.dataset_path="/home/jovyan/vmeshchaninov/HierarchicalDiffusion/data/wikipedia" \
autoencoder.latent.resolutions="[16]" \
encoder.model.mlm_probability="0.3" \
encoder.model.bert_masking="true" \
decoder.model.mlm_probabilities="[1.]" \
decoder.model.bert_masking="false" \
autoencoder.training.batch_size=256 \
autoencoder/optimizer=stableadam \
autoencoder.training.training_iters=25000
"""

"""
PROJECT_ROOT=/home/jovyan/vmeshchaninov/HierarchicalDiffusion \
torchrun --nproc_per_node=1 --master_port=12346 train_encoder.py \
dataset.name="wikipedia-bloom" \
dataset.dataset_path="/home/jovyan/vmeshchaninov/DiffusionLanguageModel/data/wikipedia-bloom" \
autoencoder.latent.resolutions="[1, 1, 2, 4]" \
encoder.model.mlm_probability="0.3" \
encoder.model.bert_masking="true" \
decoder.model.mlm_probabilities="[0.7, 0.8, 0.9, 1.]" \
decoder.model.bert_masking="true" \
autoencoder.training.batch_size=512
"""
