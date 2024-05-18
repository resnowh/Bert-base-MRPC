import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from tensorboard import program

# >>> streamlit run board.py

# 设置页面标题
st.title('基于Bert的多位置编码对比研究')

# 添加方法介绍
st.header('1. 实验设计')

st.subheader('1.1 模型选择与位置编码方案')
st.markdown('''
本实验选择了预训练的BERT模型(bert-base-uncased)作为基础模型。在此基础上,我们探索了四种不同的位置编码方案:

1. 默认的BERT位置编码(Learned Position Embedding):使用BERT原有的可学习的位置编码。
2. 屏蔽位置编码(No Position Embedding):将位置编码矩阵的权重设置为全零,相当于移除了位置编码。
3. 绝对位置编码(Absolute Position Embedding):使用固定的正弦曲线函数生成绝对位置编码,替换原有的可学习位置编码。
4. 旋转位置编码(Rotary Position Embedding):在计算注意力之前,将旋转位置编码(RoPE)应用于隐藏状态。

通过比较这四种位置编码方案,我们可以分析位置信息对于BERT在下游任务中的性能影响。
''')

st.image("D:\BaiduSyncdisk\毕业设计\参考文献\Attention Is All You Need fig1.png")

st.subheader('1.2 数据集与预处理')
st.markdown('''
我们选择了GLUE基准测试中的MRPC(Microsoft Research Paraphrase Corpus)数据集作为实验数据集。MRPC是一个句子对二分类任务,用于判断两个句子在语义上是否等价。

在数据预处理阶段,我们使用了BERT的tokenizer对句子对进行编码,将文本转换为模型可接受的输入格式。同时,我们对数据集进行了必要的截断和补零操作,以保证输入序列的长度一致。
''')

st.subheader('1.3 训练与评估')
st.markdown('''
我们使用了Hugging Face的Trainer类进行模型的训练和评估。在训练过程中,我们设置了合适的超参数,如训练轮数、批次大小、学习率等。同时,我们还定义了评估指标的计算函数,包括准确率、精确率、召回率和F1值。

为了实时监控训练过程,我们将训练日志记录到TensorBoard中,包括训练损失和评估损失的变化曲线。

在训练完成后,我们在测试集上对模型进行了评估,获取了最终的性能指标。同时,我们还可视化了位置编码的权重矩阵,以直观地观察不同位置编码方案的特点。
''')

st.subheader('1.4 实验管理与可复现性')
st.markdown('''
为了方便实验管理和结果复现,我们将不同位置编码方案的实现代码分别放置在独立的Python文件中,如PE_none.py、PE_absolute.py等。同时,我们将关键的配置参数,如位置编码类型,集中在config.py文件中进行管理。

在训练日志和可视化结果的保存路径中,我们使用了当前时间戳作为唯一标识符,确保不同运行之间的结果不会相互覆盖。同时,我们还提供了详细的代码注释和说明,方便其他研究者理解和复现我们的实验。

通过以上的实验方法,我们可以系统地比较不同位置编码方案对BERT在MRPC任务上的性能影响,并通过实验结果分析位置信息在自然语言处理任务中的作用。
''')


st.header('2. 实验数据')

# 指定TensorBoard日志目录
log_dir = './logs'

# 启动TensorBoard服务器
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

# 显示TensorBoard界面
st.markdown(f'''
    <iframe src="{url}" width="100%" height="1000" frameborder="1">
    </iframe>
''', unsafe_allow_html=True)

# 添加位置编码特点介绍
st.header('4. 使用的位置编码特点介绍')

st.subheader('空位置编码 (No Position Embedding, NoPE)')
st.markdown('''
- 不使用任何位置编码
- 完全依赖模型学习词之间的关系
- 适用于某些不依赖位置信息的任务
- 可以作为位置编码的基线比较
''')

st.subheader('学习的相对位置编码 (Learnable Relative Position Embedding)')
st.markdown('''
- Bert模型Default的位置编码
- 通过学习的方式获得位置编码
- 考虑了词之间的相对位置关系
- 能够捕捉更丰富的位置信息
- 适应性更强,可以根据任务和数据进行调整
''')

st.subheader('绝对位置编码 (Absolute Position Embedding)')
st.markdown('''
- 使用固定的三角函数生成位置编码
- 考虑了词的绝对位置信息
- 位置编码是固定的,不随训练更新
- 简单易实现,但缺乏灵活性
''')

# 绘制绝对位置编码的可视化图形
st.subheader('绝对位置编码可视化')
fig, ax = plt.subplots(figsize=(12, 12))

# 生成示例的位置索引和词嵌入维度
max_position = 128
embedding_dim = 128

# 计算绝对位置编码
position_encoding = np.zeros((max_position, embedding_dim))
position = np.arange(max_position)[:, np.newaxis]
div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
position_encoding[:, 0::2] = np.sin(position * div_term)
position_encoding[:, 1::2] = np.cos(position * div_term)

# 绘制绝对位置编码的热力图
im = ax.imshow(position_encoding, cmap='viridis', aspect='auto')
ax.set_xticks(np.arange(0, embedding_dim, 16))
ax.set_yticks(np.arange(0, max_position, 2))
ax.set_xticklabels(np.arange(0, embedding_dim, 16))
ax.set_yticklabels(np.arange(0, max_position, 2))
ax.set_xlabel('Embedding Dimension')
ax.set_ylabel('Position')
ax.set_title('Absolute Position Embedding')
fig.colorbar(im, ax=ax)

# 在Streamlit应用程序中显示图形
# st.pyplot(fig)

st.image("D:\BaiduSyncdisk\毕业设计\实验图片\PE Visual in Transformer.png")

st.markdown('''
这个图形展示了绝对位置编码的热力图表示。

- 横轴表示词嵌入的维度,纵轴表示位置索引。
- 每个位置对应一个固定的位置编码向量,不同维度的值用不同的颜色表示。
- 位置编码向量是通过三角函数(正弦和余弦)生成的,呈现出周期性的变化。

绝对位置编码通过为每个位置分配一个固定的编码向量,将位置信息融入到词嵌入中。这种编码方式简单易实现,但缺乏灵活性,因为位置编码是固定的,不会随着训练而更新。
''')

# 绘制绝对位置编码在注意力机制中的作用示意图
st.subheader('绝对位置编码在注意力机制中的作用')
fig, ax = plt.subplots(figsize=(12, 12))

# 生成示例的序列长度和注意力头数
seq_length = 128
num_heads = 4

# 计算绝对位置编码的注意力权重
position_encoding = np.zeros((seq_length, seq_length))
for i in range(seq_length):
    for j in range(seq_length):
        position_encoding[i, j] = np.sin(i / 10000 ** (2 * (j // 2) / 128)) + np.cos(i / 10000 ** (2 * (j // 2) / 128))

attention_weights = np.exp(position_encoding) / np.sum(np.exp(position_encoding), axis=-1, keepdims=True)

# 绘制注意力权重的热力图
im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')

# 设置横轴和纵轴的刻度和标签
xticks = np.arange(0, seq_length, 16)
yticks = np.arange(0, seq_length, 16)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticks, rotation=45, ha='right')
ax.set_yticklabels(yticks)

ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Absolute Position Encoding Attention Weights')
fig.colorbar(im, ax=ax)

# 在Streamlit应用程序中显示图形
st.pyplot(fig)

# 添加图形说明
st.markdown('''
这个图形展示了绝对位置编码在注意力机制中的作用,通过注意力权重的热力图来说明它如何同时提供短程和长程的注意力。

- 横轴表示键(Key)的位置,纵轴表示查询(Query)的位置。
- 热力图中的每个像素表示对应查询位置和键位置之间的注意力权重。
- 颜色越深(黄色-红色),表示注意力权重越大,即查询位置越关注相应的键位置。

从热力图中可以观察到以下特点:

1. 短程注意力:对于每个查询位置,其附近的键位置通常具有较高的注意力权重。这表明绝对位置编码能够捕捉到短程的依赖关系,使得模型更关注局部的上下文信息。

2. 长程注意力:对于距离较远的查询位置和键位置,热力图中也存在一些深色区域。这表明绝对位置编码也能够捕捉到长程的依赖关系,使得模型能够关注到序列中距离较远的信息。

3. 周期性模式:热力图呈现出一定的周期性模式,这是由于绝对位置编码使用三角函数(正弦和余弦)生成,具有周期性的特点。这种周期性模式有助于模型学习到位置信息的规律性。

综上所述,绝对位置编码通过为每个位置分配一个固定的编码向量,在注意力机制中同时提供了短程和长程的注意力。这使得模型能够捕捉到局部和全局的上下文信息,从而更好地理解和处理序列数据。
''')

st.subheader('旋转位置编码 (Rotary Position Embedding, RoPE)')
st.markdown('''
- 将位置信息编码为旋转矩阵
- 通过旋转操作将位置信息融入到词嵌入中
- 能够捕捉相对位置关系,且计算效率高
- 在自然语言处理任务中表现出色
''')
# 绘制旋转位置编码的可视化图形
st.subheader('旋转位置编码可视化')
fig, ax = plt.subplots(figsize=(12, 12))

# 生成示例的位置索引和词嵌入维度
positions = np.arange(1, 256)
embedding_dim = 512

# 计算旋转位置编码的角度
angles = positions / np.power(10000, 2 * (np.arange(embedding_dim)[:, np.newaxis] // 2) / embedding_dim)

# 绘制旋转位置编码的向量
for i, position in enumerate(positions):
    x = np.cos(angles[0, i])
    y = np.sin(angles[0, i])
    color = plt.cm.viridis(i / len(positions))
    ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc=color, ec=color, label=f'Position {position}' if i % 20 == 0 else None)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('Cosine')
ax.set_ylabel('Sine')
ax.set_title('Rotary Position Embedding')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
ax.grid(True)

# 在Streamlit应用程序中显示图形
st.pyplot(fig)

st.markdown('''
这个图形展示了旋转位置编码的可视化表示。每个箭头代表一个位置编码向量,不同的位置对应不同的旋转角度。

- 横轴表示余弦值,纵轴表示正弦值。
- 每个位置的编码向量都位于单位圆上,表示旋转的角度。
- 位置索引越大,对应的旋转角度越大,向量的方向也不同。

旋转位置编码通过将位置信息编码为旋转角度,并将其应用于词嵌入,使得模型能够捕捉词之间的相对位置关系。这种编码方式具有计算效率高和表现出色的特点。
''')

# 绘制旋转位置编码在注意力机制中的应用示意图
st.subheader('旋转位置编码在注意力机制中的应用')
fig, ax = plt.subplots(figsize=(12, 12))

# 绘制查询向量和键向量
query_vector = ax.arrow(0, 0, 0.8, 0.6, head_width=0.05, head_length=0.1, fc='r', ec='r', alpha=0.6, label='Query')
key_vector = ax.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.1, fc='g', ec='g', alpha=0.6, label='Key')

# 绘制旋转位置编码向量
position_vector = ax.arrow(0, 0, 0.6, 0.8, head_width=0.05, head_length=0.1, fc='b', ec='b', alpha=0.6, label='Position Embedding')

# 绘制应用旋转位置编码后的查询向量和键向量
rotated_query = ax.arrow(0, 0, 0.6, 1.2, head_width=0.05, head_length=0.1, fc='darkred', ec='darkred', label='Rotated Query')
rotated_key = ax.arrow(0, 0, 0.2, 1, head_width=0.05, head_length=0.1, fc='darkgreen', ec='darkgreen', label='Rotated Key')

ax.set_xlim(-0.1, 1.5)
ax.set_ylim(-0.1, 1.5)
ax.set_xlabel('Cosine')
ax.set_ylabel('Sine')
ax.set_title('Rotary Position Embedding in Attention Mechanism')
ax.legend()
ax.grid(True)

# 在Streamlit应用程序中显示图形
st.pyplot(fig)

# 添加图形说明
st.markdown('''
这个图形展示了旋转位置编码在注意力机制中的应用。

1. 红色箭头表示查询向量(Query),绿色箭头表示键向量(Key)。
2. 蓝色箭头表示旋转位置编码向量(Position Embedding)。
3. 将旋转位置编码向量分别应用于查询向量和键向量,得到旋转后的查询向量(Rotated Query,深红色箭头)和旋转后的键向量(Rotated Key,深绿色箭头)。
4. 旋转后的查询向量和键向量之间的角度差反映了它们在序列中的相对位置关系。
5. 通过计算旋转后的查询向量和键向量的点积,可以获得位置感知的注意力权重。

旋转位置编码通过将位置信息融入到查询向量和键向量中,使得注意力机制能够捕捉词之间的相对位置关系,从而提高了模型的表现。
''')

