# %%

import sys


sys.path.append("/root/steering-finetuning")

from finding_features.saes import AutoEncoderTopK

for layer in range(0,32,2):
    sae = AutoEncoderTopK.from_pretrained(layer)
