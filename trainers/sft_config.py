from pydantic.dataclasses import dataclass

@dataclass
class SFTConfig:
    ### WANDB ARGS ###

    wb_project: str

    wb_run_name: str

    wb_run_group: str

    ### TRAINING ARGS ###

    seed: int

    batch_size: int

    eval_batch_size: int

    epochs: int

    lr: float

    warmup_ratio: float

    per_device_batch_size: int

    @property
    def acc_steps(self):
        if self.batch_size % self.per_device_batch_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by per_device_batch_size {self.per_device_batch_size}"
            )
        return self.batch_size // self.per_device_batch_size

    @property
    def wb_config(self):
        if self.seed is None:
            raise ValueError("Seed must be set")

        return {
            "batch_size": self.batch_size,
            "per_device_batch_size": self.per_device_batch_size,
            "acc_steps": self.acc_steps,
            "epochs": self.epochs,
            "seed": self.seed,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "wb_run_group": self.wb_run_group,
        }
