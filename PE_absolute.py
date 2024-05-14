import torch


def convert_to_absolute_position_embedding(model, max_length=512, embedding_dim=768):
    # 获取模型的位置编码权重
    position_embeddings = model.bert.embeddings.position_embeddings.weight.detach().cpu().numpy()

    # 创建新的绝对位置编码矩阵
    absolute_position_embedding = torch.zeros(max_length, embedding_dim)

    # 计算三角函数的频率
    freq_seq = torch.arange(0, embedding_dim, 2.0) / embedding_dim
    inv_freq = 1 / (10000 ** freq_seq)

    # 生成绝对位置编码
    sinusoid_input = torch.ger(torch.arange(max_length), inv_freq)
    absolute_position_embedding[:, 0::2] = torch.sin(sinusoid_input)
    absolute_position_embedding[:, 1::2] = torch.cos(sinusoid_input)

    # 将绝对位置编码复制到模型的位置编码中
    model.bert.embeddings.position_embeddings.weight.data.copy_(absolute_position_embedding)

    return model
