import matplotlib.pyplot as plt
import torch

from utils import VisualTrainer

def train_and_evaluate(model, tokenizer, dataset, training_args):
    # 将模型移动到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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