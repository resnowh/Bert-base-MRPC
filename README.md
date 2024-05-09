# BERT-BASE-MRPC

使用GLUE基准测试中的MRPC数据集, 训练bert-base-uncased模型.

## 实时可视化: 使用TensorBoard实现训练指标的实时可视化
1. 安装TensorBoard: `pip install tensorboard`

2. 配置TrainingArguments: 
    ```python
    # 添加以下参数以启用TensorBoard日志记录:  
    training_args = TrainingArguments(
        #...
        logging_dir='./logs',            # TensorBoard日志目录
        logging_steps=100,               # 每100步记录一次日志
        #...
    )
    ```
3. 启动TensorBoard: 在训练前, 于项目根目录下运行`tensorboard --logdir=./logs`

4. 访问TensorBoard界面: http://localhost:6006

5. 开始训练, 并查看实时的可视化信息

## 使用GPU训练模型

1. 安装GPU版本的PyTorch: 访问 https://pytorch.org/get-started/locally/ , 获取下载安装命令

2. 将模型移动到GPU上: 在创建模型后,使用`model.to(device)`将模型移动到GPU设备上。例如:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
   model.to(device)
   ```

3. 将数据移动到GPU上: 在训练循环中,确保将输入数据移动到与模型相同的设备上。例如:
   ```python
   for batch in train_dataloader:
       inputs = {k: v.to(device) for k, v in batch.items()}
       outputs = model(**inputs)
       ...
   ```

4. 配置Trainer以使用GPU: 如果你使用Hugging Face的`Trainer`类进行训练,可以通过设置`TrainingArguments`中的`device`参数来指定使用的设备。例如:
   ```python
   training_args = TrainingArguments(
       ...
       device='cuda',  # 指定使用GPU
       ...
   )
   ```

5. 检查GPU使用情况: 在训练过程中, 使用`nvidia-smi`命令来检查GPU的使用情况, 确保模型训练在利用GPU资源。
