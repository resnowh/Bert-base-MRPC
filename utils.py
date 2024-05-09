from transformers import Trainer
from torch.utils.tensorboard import SummaryWriter


class VisualTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)

    def log(self, logs):
        super().log(logs)
        for k, v in logs.items():
            self.writer.add_scalar(k, v, self.state.global_step)