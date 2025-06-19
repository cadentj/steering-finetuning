# %%

import sys

sys.path.append("/root/steering-finetuning")

# %%

from finding_features.saes import AutoEncoderTopK


print(list(range(0, 26, 2)))
saes = [AutoEncoderTopK.from_pretrained(i) for i in range(0, 26, 2)]

# %%

saes[0]