import streamlit as st
from tensorboard import program

# >>> streamlit run board.py

# 设置页面标题
st.title('基于Bert-base-uncased模型的不同位置编码对比')

# 指定TensorBoard日志目录
log_dir = './logs'

# 启动TensorBoard服务器
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

# 显示TensorBoard界面
st.markdown(f'''
    <iframe src="{url}" width="100%" height="1000" frameborder="0">
    </iframe>
''', unsafe_allow_html=True)
