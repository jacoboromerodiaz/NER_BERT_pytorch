from transformers import BertTokenizer, BertModel

TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL = BertModel.from_pretrained("bert-base-cased", return_dict=False)
TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")
MODEL_PATH = '.\output\model.pth'
