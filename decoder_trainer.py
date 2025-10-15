# Python standard library
from typing import Dict, Tuple, Union

# Third party libraries
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import trange
from transformers import AutoTokenizer
from encoder_trainer import EncoderTrainer
from encoder_trainer import cross_entropy_loss

from architecture.encoder import Encoder
from architecture.decoder import Decoder

from utils.logging_utils import config_to_wandb

from encoder_trainer import mse_loss_function, accuracy


class DecoderTrainer(EncoderTrainer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.step = 0

        # Initialize tokenizer and set vocab configs
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.autoencoder.model.text_encoder)
        self.vocab_size = self.tokenizer.vocab_size

        self.device = torch.device(f"cuda:{self.cfg.ddp.local_rank}") if self.cfg.ddp.enabled else torch.device("cuda")
        
        # Configure encoder
        self._setup_encoder_cfg()
        self.encoder = Encoder(self.cfg.encoder).cuda()
        # Freezing encoder parameters
        for par in self.encoder.parameters():
            par.requires_grad = False
        
        # Configure decoder cfg
        self._setup_decoder_cfg()
        self.decoder = Decoder(self.cfg.decoder).cuda()

        self.restore_checkpoint()

        if self.cfg.training == "autoencoder":
            # Initialize training components
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_grad_scaler()
        
            # Setup DDP
            if dist.is_initialized() and self.cfg.ddp.enabled:
                self._setup_ddp()

            # Log parameter counts
            self._log_parameter_counts()
            
            if dist.is_initialized() and dist.get_rank() == 0:
                config_to_wandb(self.cfg)
    
    def _setup_ddp(self):
        """Setup Distributed Data Parallel."""
        self.ddp_encoder = self.encoder
        self.ddp_decoder = self.decoder
        
        if self.cfg.ddp.enabled:  
            self.ddp_decoder = torch.nn.parallel.DistributedDataParallel(
                self.decoder,
                device_ids=[self.cfg.ddp.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def train(self) -> None:
        self.train_range = trange(self.step + 1, self.cfg.decoder.finetuning.training_iters + 1)
        self.train_loader_iter = iter([])

        self.ddp_encoder.eval()
        self.ddp_decoder.train()

        for step in self.train_range:
            self.step = step
            
            batch = next(self.train_loader_iter, None)
            if batch is None:
                self._setup_train_data_generator()
                self.train_loader_iter = iter(self.train_loader)
                batch = next(self.train_loader_iter, None)

            total_loss, loss_dict = self.calc_loss(batch)
            stat_dict = self.optimizer_step(total_loss)

            if self.step % self.cfg.autoencoder.logging.log_freq == 0:
                if dist.is_initialized() and dist.get_rank() == 0:
                    self.log_data(total_loss, loss_dict, stat_dict, is_train=True)   
            
            self.train_range.set_description(f"total_loss: {total_loss.item():0.3f}")
            
            if self.step % self.cfg.autoencoder.logging.save_freq == 0:
                self.latent_mean, self.latent_std = self.get_latent_statistics()
                self.save_checkpoint()

                self.ddp_encoder.eval()
                self.ddp_decoder.train()

            if self.step % self.cfg.autoencoder.logging.eval_freq == 0:
                self.validate()
                torch.cuda.empty_cache()

                self.ddp_encoder.eval()
                self.ddp_decoder.train()

        self.save_checkpoint()

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor]]:
        batch = batch.to(self.device)
        
        encoder_latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=self.cfg.decoder.finetuning.bert_output_masking)
        encoder_latents = self.normalize_latent(encoder_latents)

        # Gaussian noise
        eps = torch.randn_like(encoder_latents)
        alpha = self.cfg.decoder.finetuning.alpha + (1 - self.cfg.decoder.finetuning.alpha) * torch.rand(encoder_latents.shape[0], device=self.device)
        #torch.ones(encoder_latents.shape[0], device=self.device)
        sigma = torch.sqrt(1 - alpha ** 2)
        if self.cfg.decoder.finetuning.is_alpha:
            latents = alpha[..., None, None] * encoder_latents + sigma[..., None, None] * eps
        else:
            latents = encoder_latents + sigma[..., None, None] * eps

        # Masking
        if self.cfg.decoder.finetuning.latent_masking:
            p = self.cfg.encoder.augmentation.latent_masking.probability
            latents = latents * (torch.rand_like(latents) > p)

        latents = self.denormalize_latent(latents)

        # get decoder logits
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, hidden_state_of_decoder = self.ddp_decoder(
                encoder_latents=latents, 
                return_last_hidden_state=True,
            )

        # Compute loss for decoder
        ce_loss = cross_entropy_loss(
            input=logits,
            target=batch["input_ids"],
            mask=batch["attention_mask"]
        )

        mse_loss = mse_loss_function(
            input=hidden_state_of_decoder,
            target=bert_hidden_state.detach().clone(),
            mask=batch["attention_mask"],
        )

        total_loss = ce_loss + mse_loss
        
        # Logging
        stat_dict = {}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            stat_dict["ce_loss"] = ce_loss.detach().item()
            stat_dict["mse_loss"] = mse_loss.detach().item()

            acc = accuracy(
                logits=logits,
                target=batch["input_ids"],
                mask=batch["attention_mask"]
            )
            stat_dict["accuracy"] = acc.detach().item()

        return total_loss, stat_dict