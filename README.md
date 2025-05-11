# Steering Finetuning

# Installation

Set up the repository by running `uv pip install -r requirements`.

# Reproducing Experiments

## Section 5 - Controlling Emergent Misalignment

Train the models from scripts in

## Section 6 - Steering in Multiple Choice Settings

Train the initial models from `scripts/train_mcmc_pca.sh --type base` and `scripts/train_gender_pca.sh --type base`

Save PCs with `scripts/make_mcmc_pca.sh` and `scripts/make_mcmc_pca.sh`

Interpret the PCs with the notebooks and utils in `finding_features`. After saving relevant feature dictionaries to the `finding_features/features` directory, generate intervention dictionaries with `finding_features/generate_interventions.py`.

Train models with interventions using `scripts/train_mcmc_pca.sh --type intervention`. Run the relevant baselines with `scripts/train_mcmc_pca.sh --type [random|top|test_time]`.

Download run data from `wandb`, then create relevant plots using the scripts in results. Plotting scripts are prepended with their figure number.