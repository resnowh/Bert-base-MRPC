from model import load_model_and_tokenizer
from dataset import load_and_preprocess_dataset
from train import train_and_evaluate


def main():
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 加载和预处理数据集
    dataset = load_and_preprocess_dataset(tokenizer)

    # 训练和评估模型
    while True:
        try:
            train_and_evaluate(model, tokenizer, dataset)
            break  # 如果训练成功完成,跳出循环
        except KeyboardInterrupt:
            print("Training interrupted. Resuming from the latest checkpoint...")
            continue  # 如果训练被中断,继续下一次循环


if __name__ == '__main__':
    main()
