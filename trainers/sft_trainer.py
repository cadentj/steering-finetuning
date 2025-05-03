import wandb as wb
import torch as t
from torch.utils.data import DataLoader
from nnsight import LanguageModel
from transformers import get_scheduler
from tqdm import tqdm

from .config import SFTConfig

def stuff():
    return SFTConfig


# class SFTHarness(Harness):
#     def __init__(
#         self, 
#         model: LanguageModel, 
#         trainer: Trainer,
#         train_data: DataLoader,
#         val_data: DataLoader,
#         test_data: DataLoader,
#         cfg: SFTConfig,
#     ):
#         assert model.tokenizer.padding_side == "left", "Padding side must be left"


#         print(f"Initializing wandb with project: {cfg.wb_project}")
#         self.run = wb.init(
#             project=cfg.wb_project,
#             name=cfg.wb_run_name,
#             config=asdict(cfg)
#         )
#         print(f"Wandb run initialized with: {self.run.project}")

#         self.cfg = cfg

#         self.model = model
#         self.trainer = trainer
#         self.train_data = train_data
#         self.val_data = val_data
#         self.test_data = test_data
#         super().__init__(cfg)

#     def train(self):
#         n_steps = len(self.train_data) * self.cfg.epochs
#         n_warmup_steps = int(n_steps * self.cfg.warmup_ratio)

#         if self.cfg.optim == "sgd":
#             optim = t.optim.SGD(
#                 self.model.parameters(), 
#                 lr=self.cfg.lr,
#                 momentum=0.9,
#                 weight_decay=0.01
#             )
#         elif self.cfg.optim == "adamw":
#             optim = t.optim.AdamW(
#                 self.model.parameters(),
#                 lr=self.cfg.lr,
#                 weight_decay=0.01,
#                 betas=(0.9, 0.95)
#             )
#         else:
#             raise ValueError(f"Unknown optimizer: {self.cfg.optim}")
        
#         lr_scheduler = get_scheduler(
#             "linear",
#             optim,
#             num_warmup_steps=n_warmup_steps,
#             num_training_steps=n_steps
#         )

#         pbar = tqdm(range(n_steps // self.cfg.acc_steps))
#         optim.zero_grad()
#         self.validate(use_val_data=True)
        
#         self.model.train()
#         for epoch in range(self.cfg.epochs):
#             for it, x in enumerate(self.train_data):
#                 with self.model.trace(
#                     x['formatted'], 
#                     use_cache=False,
#                     invoker_args={"truncation" : True, "max_length" : 512}
#                 ):
#                     loss, acc = self.trainer.step(x, intervention=True)
#                     loss = loss / self.cfg.acc_steps

#                     loss = loss.save()
#                     acc = acc.save()

#                     loss.backward()

#                 if (it + 1) % self.cfg.acc_steps == 0:
#                     wb.log({
#                         "train/loss": (loss * self.cfg.acc_steps).item(),
#                         "train/accuracy": acc.item(),
#                         "train/learning_rate": optim.param_groups[0]["lr"]
#                     })

#                     optim.step()
#                     lr_scheduler.step()
#                     optim.zero_grad()
#                     pbar.update(1)

#                 # Validate every 1/4 epoch
#                 if (it + 1) % (len(self.train_data) // 4) == 0:
#                     self.validate(use_val_data=False)

#             # Validate at the end of each epoch
#             self.validate(use_val_data=True)

#         wb.finish()

#     def validate(self, use_val_data=False):
#         self.model.eval()
        
#         with t.no_grad():
#             # Use validation data
#             if use_val_data:
#                 loss = []
#                 acc = []
#                 loss_intervention = []
#                 acc_intervention = []

#                 for it, x in enumerate(self.val_data):
#                     with self.model.trace(x['formatted'], use_cache=False):
                        
#                         # loss/acc without intervention
#                         _loss, _acc = self.trainer.step(x, intervention=False)
#                         loss.append(_loss.save())
#                         acc.append(_acc.save())
                        
#                     with self.model.trace(x['formatted'], use_cache=False):
#                         # loss/acc with intervention
#                         _loss_intervention, _acc_intervention = self.trainer.step(x, intervention=True)
#                         loss_intervention.append(_loss_intervention.save())
#                         acc_intervention.append(_acc_intervention.save())
        
#                 wb.log({
#                     "val/loss": t.stack(loss).mean(),
#                     "val/accuracy": t.stack(acc).mean(),
#                     "val/intervention/loss": t.stack(loss_intervention).mean(),
#                     "val/intervention/accuracy": t.stack(acc_intervention).mean()
#                 }, commit=False)

#             # Use test data 
#             loss = []
#             acc = []
#             loss_flipped = []
#             acc_flipped = []
#             loss_intervention = []
#             acc_intervention = []
#             loss_flipped_intervention = []
#             acc_flipped_intervention = []

#             for it, x in enumerate(self.test_data):
#                 with self.model.trace(x['formatted'], use_cache=False):     
#                     # loss/acc without intervention
#                     _loss, _acc = self.trainer.step(x, intervention=False)
#                     loss.append(_loss.save())
#                     acc.append(_acc.save())

#                     # loss/acc with flipped answer
#                     _loss_flipped, _acc_flipped = self.trainer.step(x, intervention=False, flip_answer=True)
#                     loss_flipped.append(_loss_flipped.save())
#                     acc_flipped.append(_acc_flipped.save())
                
#                 with self.model.trace(x['formatted'], use_cache=False):
#                     # loss/acc with intervention
#                     _loss_intervention, _acc_intervention = self.trainer.step(x, intervention=True)
#                     loss_intervention.append(_loss_intervention.save())
#                     acc_intervention.append(_acc_intervention.save())

#                     # loss/acc with flipped answer and intervention
#                     _loss_flipped_intervention, _acc_flipped_intervention = self.trainer.step(x, intervention=True, flip_answer=True)
#                     loss_flipped_intervention.append(_loss_flipped_intervention.save())
#                     acc_flipped_intervention.append(_acc_flipped_intervention.save())
    
#             wb.log({
#                 "test/loss": t.stack(loss).mean(),
#                 "test/accuracy": t.stack(acc).mean(),
#                 "test/loss_flipped": t.stack(loss_flipped).mean(),
#                 "test/accuracy_flipped": t.stack(acc_flipped).mean(),
#                 "test/intervention/loss": t.stack(loss_intervention).mean(),
#                 "test/intervention/accuracy": t.stack(acc_intervention).mean(),
#                 "test/intervention/loss_flipped": t.stack(loss_flipped_intervention).mean(),
#                 "test/intervention/accuracy_flipped": t.stack(acc_flipped_intervention).mean()
#             }, commit=False)



        

#         self.model.train()