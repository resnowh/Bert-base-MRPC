from transformers import Trainer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import collections
from tqdm import tqdm


def compute_metrics(pred):
    # 获取真实标签
    labels = pred.label_ids

    # 获取预测结果
    # 如果 predictions 是概率分布,使用 argmax 函数获取概率最大的类别索引作为预测结果
    # 如果 predictions 是 logits,argmax 函数会返回 logits 最大的类别索引作为预测结果
    preds = pred.predictions.argmax(-1)

    # 计算准确率
    acc = accuracy_score(labels, preds)

    # 计算精确率
    precision = precision_score(labels, preds)

    # 计算召回率
    recall = recall_score(labels, preds)

    # 计算 F1 值
    f1 = f1_score(labels, preds)

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(labels, preds).tolist()

    # 返回评估指标的字典
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_mat
    }


class VisualTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)

    def log(self, logs):
        super().log(logs)
        for k, v in logs.items():
            if k == "loss":
                # 将训练损失记录到 TensorBoard
                self.writer.add_scalar("Loss/Train", v, self.state.global_step)
            elif k == "eval_loss":
                # 将评估损失记录到 TensorBoard
                self.writer.add_scalar("Loss/Eval", v, self.state.global_step)

