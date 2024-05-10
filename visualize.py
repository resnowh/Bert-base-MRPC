import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_position_embeddings(model, tokenizer, max_length=128):
    # 获取位置编码的权重
    position_embeddings = model.bert.embeddings.position_embeddings.weight.detach().cpu().numpy()

    # 绘制热力图
    plt.figure(figsize=(10, 10))
    sns.heatmap(position_embeddings[:max_length], cmap='coolwarm', center=0, robust=True, annot=False, cbar_kws={'label': 'Position Embedding Value'})
    plt.title('Position Embeddings')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.savefig('position_embeddings.png', dpi=300)
    plt.show()
