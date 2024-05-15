def remove_position_embeddings(model):
    # 将位置编码权重设置为全零
    model.bert.embeddings.position_embeddings.weight.data.zero_()

    # 冻结位置编码权重,确保在训练过程中不会被更新
    model.bert.embeddings.position_embeddings.weight.requires_grad = False

    return model
