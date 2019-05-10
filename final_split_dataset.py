import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# TEST_SIZE = 0.05

# MAX_LEN = 100

train_de = open('./data/iwslt16.moses.tokenized.16k.final.de-en/train.de', encoding='utf-8').read().split('\n')
train_en = open('./data/iwslt16.moses.tokenized.16k.final.de-en/train.en', encoding='utf-8').read().split('\n')

dev_de = open('./data/iwslt16.moses.tokenized.16k.final.de-en/dev.de', encoding='utf-8').read().split('\n')
dev_en = open('./data/iwslt16.moses.tokenized.16k.final.de-en/dev.en', encoding='utf-8').read().split('\n')

test_de = open('./data/iwslt16.moses.tokenized.16k.final.de-en/test.de', encoding='utf-8').read().split('\n')

raw_data_train = {'English' : [line for line in train_en], 'German': [line for line in train_de]}
raw_data_dev = {'English' : [line for line in dev_en], 'German': [line for line in dev_de]}
raw_data_test = {'English' : ['' for line in test_de], 'German' : [line for line in test_de]}


df_train = pd.DataFrame(raw_data_train, columns=["German", "English"])
df_val = pd.DataFrame(raw_data_dev, columns=["German", "English"])
df_test = pd.DataFrame(raw_data_test, columns=["German", "English"])

df_train['en_len'] = df_train['English'].str.count(' ')
df_train['de_len'] = df_train['German'].str.count(' ')
df_val['en_len'] = df_train['English'].str.count(' ')
df_val['de_len'] = df_train['German'].str.count(' ')
df_test['en_len'] = df_test['English'].str.count(' ')
df_test['de_len'] = df_test['German'].str.count(' ')

# df_train = df_train.query('de_len < {} & en_len < {}'.format(MAX_LEN, MAX_LEN))
# df_test = df_test.query('de_len < {} & en_len < {}'.format(MAX_LEN, MAX_LEN))

# train, val = train_test_split(df_train, test_size=TEST_SIZE)

df_train.to_csv("./data/train_fin.csv", index=False)
df_val.to_csv("./data/val_fin.csv", index=False)
df_test.to_csv("./data/test_fin.csv", index=False)



