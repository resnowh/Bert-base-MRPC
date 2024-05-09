from transformers import BertTokenizer, BertForSequenceClassification


def load_model_and_tokenizer():
    # 加载预训练的BERT tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    return model, tokenizer
