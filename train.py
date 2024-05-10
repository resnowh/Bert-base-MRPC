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
        num_train_epochs=4,  # 训练轮数
        per_device_train_batch_size=4,  # 训练
        per_device_eval_batch_size=4,  # 测试
        warmup_steps=50,  # 预热步数
        learning_rate=2e-5,  # 学习率
        weight_decay=0.01,  # 权重衰减系数
        gradient_accumulation_steps=8,  # 梯度累积步数
        # 导出日志
        logging_dir='./logs',  # 日志路径
        logging_steps=20,  # 日志间隔
        # 检查点
        evaluation_strategy='steps',  # 评估策略
        eval_steps=50,  # 评估间隔
        save_strategy='steps',  # 保存策略
        save_steps=10,  # 保存间隔
        save_total_limit=5,  # 最多保存5个最新的检查点
        load_best_model_at_end=False,  # 是否在训练结束时加载最佳模型（开启后需要保证eval_strategy和save_strategy保持一致）
    )

    # 初始化自定义的Trainer
    trainer = VisualTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval']
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=True)  # 重新开始训练时应将resume_from_checkpoint设置为False

    # 在测试集上评估模型
    eval_results = trainer.evaluate(eval_dataset=dataset['eval'])
    print(eval_results)

    # 可视化训练过程中的损失函数变化
    plt.figure(figsize=(10, 6))

    # 绘制训练集loss
    train_loss_vals = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    train_steps = [log['step'] for log in trainer.state.log_history if 'loss' in log]
    plt.plot(train_steps, train_loss_vals, label='Training Loss')

    # 绘制验证集loss
    eval_loss_vals = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]
    plt.plot(eval_steps, eval_loss_vals, label='Validation Loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    # 保存模型
    trainer.save_model('./trained_model')
