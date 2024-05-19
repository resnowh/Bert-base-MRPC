from transformers import BertTokenizer, BertForSequenceClassification
from PE_sinusoidal import convert_to_sinusoidal_position_embedding
from PE_none import remove_position_embeddings
from PE_relative import convert_to_relative_position_embedding
from PE_rotary import convert_to_rotary_position_embedding
from config import position_embedding_type


def load_model_and_tokenizer():
    # 加载预训练的BERT tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # 默认位置编码
    if position_embedding_type == 'absolute':
        model = model
    elif position_embedding_type == 'relative_key':
        model = model
    elif position_embedding_type == 'relative_key_query':
        model = model

    # 屏蔽位置编码
    elif position_embedding_type == 'none':
        model = remove_position_embeddings(model)

    # 三角位置编码
    elif position_embedding_type == 'sinusoidal':
        model = convert_to_sinusoidal_position_embedding(model)

    # 旋转位置编码
    elif position_embedding_type == 'rotary':
        model = convert_to_rotary_position_embedding(model)

    else:
        raise ValueError(f"Unsupported position embedding type: {position_embedding_type}")

    return model, tokenizer
