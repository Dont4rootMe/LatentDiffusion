# PROJECT_ROOT=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/LatentDiffusion
# export PROJECT_ROOT

PYTHONPATH=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/LatentDiffusion \
    accelerate launch \
    --config_file /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/accelerate_config.yaml \
    train_encoder.py \
    project.run_name='recheck_on_one_latent_vae' \
    decoder.latent.num_latents=1 \
    encoder.latent.num_latents=1 \

