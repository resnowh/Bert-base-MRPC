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
        num_train_epochs=5,  # 训练轮数
        per_device_train_batch_size=4,  # 训练
        per_device_eval_batch_size=4,  # 测试
        warmup_steps=500,  # 预热步数
        weight_decay=0.01,  # 权重衰减系数
        gradient_accumulation_steps=4,  # 梯度累积步数
        # 导出日志
        logging_dir='./logs',  # 日志路径
        logging_steps=20,  # 日志间隔
        # 检查点
        save_strategy='steps',  # 添加保存策略为每隔一定步数保存一次
        save_steps=10,  # 每隔10步保存一次检查点
        save_total_limit=5,  # 最多保存5个最新的检查点
        load_best_model_at_end=False,  # 是否在训练结束时加载最佳模型（开启后需要保证eval_strategy和save_strategy保持一致）
    )

    # 初始化自定义的Trainer
    trainer = VisualTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train']
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=True)

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
