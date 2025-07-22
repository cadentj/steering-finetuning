Replication scripts

# Section 4: Controlling Emergent Misalignment 

Run `compute_sae_latents.sh` to get SAE latents for Qwen and Mistral.

Run `compute_pca_latents.sh` to get PCA latents.

Run `interpret.sh` to compute feature pages for the latents.

Run `train.sh` to train with interventions.

# Section 5: Reducing Sensitivity to Spurious Cues

Run `compute_sae_latents.sh` to get SAE latents for Gemma.

Run `compute_pca_latents.sh` to get PCA latents.

The above will compute latents for all MCMC combinations specified in `spurious_configs.yaml` and gender latents as well.

Run `interpret.sh` to compute feature pages for the latents.

Run `train.sh` to train with interventions.
