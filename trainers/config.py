from pydantic import BaseModel

class SFTConfig(BaseModel):
    ### WANDB ARGS ###

    wb_project: str

    wb_run_name: str

    wb_run_group: str

    use_wb: bool = True

    ### TRAINING ARGS ###

    save_dir: str = None

    batch_size: int = 16

    eval_batch_size: int = 32

    epochs: int = 2

    lr: float = 2e-5

    warmup_ratio: float = 0.15

    per_device_batch_size: int = 4

    seed: int = 42

    @property
    def acc_steps(self):
        if self.batch_size % self.per_device_batch_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by per_device_batch_size {self.per_device_batch_size}"
            )
        return self.batch_size // self.per_device_batch_size

    @property
    def wb_config(self):
        return {
            "batch_size": self.batch_size,
            "per_device_batch_size": self.per_device_batch_size,
            "acc_steps": self.acc_steps,
            "epochs": self.epochs,
            "seed": self.seed,
            "optim": self.optim,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "wb_run_group": self.wb_run_group,
            "pefted" : self.peft_config is not None,
        }
