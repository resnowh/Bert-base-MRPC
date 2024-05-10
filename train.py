import matplotlib.pyplot as plt
import torch
from config import training_args
from utils import VisualTrainer
from visualize import visualize_position_embeddings, plot_loss


def train_and_evaluate(model, tokenizer, dataset):
    # 将模型移动到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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

    # 可视化位置编码
    visualize_position_embeddings(model, tokenizer)

    # 提取训练和验证的loss值和步数
    train_loss_vals = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    train_steps = [log['step'] for log in trainer.state.log_history if 'loss' in log]
    eval_loss_vals = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]

    # 可视化训练过程中的损失函数变化
    plot_loss(train_loss_vals, train_steps, eval_loss_vals, eval_steps)

    # 保存模型
    trainer.save_model('./trained_model')
