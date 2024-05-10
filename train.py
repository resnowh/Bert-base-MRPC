import matplotlib.pyplot as plt
import torch
from transformers import TrainingArguments

from utils import VisualTrainer


def train_and_evaluate(model, tokenizer, dataset):
    # 将模型移动到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',  # 输出目录
        num_train_epochs=3,  # 训练轮数
        per_device_train_batch_size=4,  # 训练
        per_device_eval_batch_size=4,  # 测试
        warmup_steps=500,  # 预热步数
        weight_decay=0.01,  # 权重衰减系数
        logging_dir='./logs',  # 日志路径
        logging_steps=20,  # 日志间隔
        gradient_accumulation_steps=4,  # 梯度累积步数
    )

    # 初始化自定义的Trainer
    trainer = VisualTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train']
    )

    # 开始训练
    trainer.train()

    # 在测试集上评估模型
    eval_results = trainer.evaluate(eval_dataset=dataset['eval'])
    print(eval_results)

    # 可视化训练过程中的损失函数变化
    plt.figure(figsize=(10, 6))
    loss_vals = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    steps = [log['step'] for log in trainer.state.log_history if 'loss' in log]
    plt.plot(steps, loss_vals)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    # 保存模型
    trainer.save_model('./trained_model')
