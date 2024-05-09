from model import load_model_and_tokenizer
from dataset import load_and_preprocess_dataset
from train import train_and_evaluate
from transformers import TrainingArguments


def main():
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 加载和预处理数据集
    dataset = load_and_preprocess_dataset(tokenizer)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',         # 输出目录
        num_train_epochs=3,             # 训练轮数
        per_device_train_batch_size=4,  # 训练
        per_device_eval_batch_size=4,   # 测试
        warmup_steps=500,               # 预热步数
        weight_decay=0.01,              # 权重衰减系数
        logging_dir='./logs',           # 日志路径
        logging_steps=20,               # 日志间隔
        gradient_accumulation_steps=4,  # 梯度累积步数
    )

    # 训练和评估模型
    train_and_evaluate(model, tokenizer, dataset, training_args)


if __name__ == '__main__':
    main()
