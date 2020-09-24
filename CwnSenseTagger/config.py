import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#TEST_DATA = os.path.join(PROJECT_ROOT, "data/train_2008011514_data.json")
#TEST_JSON = os.path.join(PROJECT_ROOT, "test/test.json")

BERT_MODEL = "bert-base-chinese"
MODEL_DIR = ""
PAD = 0
UNK = 1
CLS = 2
SEP = 3
COMMA = 117
LESS_THAN = 133
LARGER_THAN = 135

