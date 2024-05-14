# BERT-BASE-MRPC

使用GLUE基准测试中的MRPC数据集, 训练bert-base-uncased模型.

## Todo-List
- [x] 可视化监控训练参数(TensorBoard)
- [x] 实现训练从中断点继续
- [x] 训练完成后绘制loos和位置编码图像
- [x] 实现模型性能评估
  - [x] 准确率、精确率、召回率、F1值和混淆矩阵
- [ ] 修改bert模型的学习型位置编码，实现
  - [ ] 学习型PE（默认）
  - [ ] 绝对PE
  - [ ] 相对PE
  - [ ] RoPE
  - [ ] NoPE
- [ ] 使用不同数据集分别测试
  - [ ] GLUE-MRPC（Microsoft Research Paraphrase Corpus）

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

## ~~修改评估进度条~~
修改了trainer.py中的 `evaluation_loop` 函数。

在原有的函数中,进度条是通过直接遍历 `dataloader` 来创建的:

```python
for step, inputs in enumerate(dataloader):
    ...
```

将这部分代码替换为使用 tqdm 进度条的逻辑:

```python
steps = len(dataloader)
progress_bar = tqdm(dataloader, total=steps, disable=not self.is_local_process_zero() or self.args.disable_tqdm, leave=False, desc=description)

for step, inputs in enumerate(progress_bar):
    ...
```

这样,进度条将使用 tqdm 库来创建和更新。

另外,在函数的最后,可以添加一行代码来关闭进度条:

```python
progress_bar.close()
```

这将确保在评估循环结束后,进度条被正确地关闭。

`tqdm` 是一个用于创建进度条的第三方库,需要通过 `import` 语句导入后才能使用。你需要在文件的开头添加以下导入语句:

```python
from tqdm import tqdm
```