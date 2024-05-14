import os
import shutil

import matplotlib.pyplot as plt
import torch
from config import training_args
from utils import VisualTrainer, compute_metrics
from visualize import visualize_position_embeddings, plot_loss


def train(model, tokenizer, dataset):
    trainer = VisualTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        compute_metrics=compute_metrics
    )
    trainer.train(resume_from_checkpoint=False)
    return trainer


def evaluate(model, tokenizer, dataset, trainer):
    # 在测试集上评估模型
    eval_results = trainer.evaluate(eval_dataset=dataset['test'])
    print("Evaluation results:")
    print(eval_results)

    # 可视化位置编码
    visualize_position_embeddings(model, tokenizer)

    # 提取训练和验证的loss值和步数
    train_loss_vals = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    train_steps = [log['step'] for log in trainer.state.log_history if 'loss' in log]
    eval_loss_vals = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]

    # 可视化训练过程中的损失函数变化
    plot_loss(train_loss_vals, train_steps, eval_loss_vals, eval_steps)


def train_and_evaluate(model, tokenizer, dataset):
    # 将模型移动到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 删除已有的TensorBoard日志目录
    log_dir = os.path.join(training_args.logging_dir, training_args.run_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # 训练模型
    trainer = train(model, tokenizer, dataset)

    # 评估模型
    evaluate(model, tokenizer, dataset, trainer)

    # 保存模型
    trainer.save_model('./trained_model')
