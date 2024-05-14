from absolute_PE import convert_to_absolute_position_embedding
from config import position_embedding_type
from model import load_model_and_tokenizer
from dataset import load_and_preprocess_dataset
from train import train_and_evaluate


def main():
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 根据配置文件中的关键字决定是否将位置编码转换为绝对位置编码
    if position_embedding_type == 'absolute':
        model = convert_to_absolute_position_embedding(model)

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
