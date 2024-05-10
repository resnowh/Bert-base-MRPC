from model import load_model_and_tokenizer
from dataset import load_and_preprocess_dataset
from train import train_and_evaluate


def main():
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 加载和预处理数据集
    dataset = load_and_preprocess_dataset(tokenizer)

    # 训练和评估模型
    train_and_evaluate(model, tokenizer, dataset)


if __name__ == '__main__':
    main()
