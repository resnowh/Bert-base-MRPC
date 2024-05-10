from transformers import TrainingArguments

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 输出目录
    num_train_epochs=4,  # 训练轮数
    per_device_train_batch_size=4,  # 训练批次大小
    per_device_eval_batch_size=4,  # 测试批次大小
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
