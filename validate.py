import logging
import spacy
import torch
import sys
import torch.nn as nn
from torchtext import data, datasets
from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from transformer.my_iterator import MyIterator, rebatch
WRITE_FILE = './translations/transformer_baseline.txt'
TARGET_FILE = './translations/transformer_baseline_tgt.txt'
LOG_FILE = './log/logged.log'
logging.basicConfig(filename= LOG_FILE, filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

SAVE_DIR = './saved_models/'
MODEL_FILE = 'transformer_baseline_1000.pt'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
BATCH_SIZE = 1000


def tokenize_de(text):
    #return text.split()
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    #return text.split()
    return [tok.text for tok in spacy_en.tokenizer(text)]

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


save_dict = torch.load(SAVE_DIR+MODEL_FILE)
SRC = save_dict['SRC']
TGT = save_dict['TGT']
model = make_model(len(SRC.vocab), len(TGT.vocab))
model.load_state_dict(save_dict['model_state_dict'])
model.cuda()
model.eval()
logging.info('Model Loaded!')
data_fields = [('src', SRC), ('trg', TGT)]

train, val, test = data.TabularDataset.splits(path='./data', train='train_1000.csv', validation='val_1000.csv', test='test_1000.csv', format='csv', fields=data_fields, skip_header=True)

test_iter = data.BucketIterator(test, batch_size=1, device=0, shuffle=False, sort=None)
logging.info('Started Decoding process!')
write_file = open(WRITE_FILE, "w+")
target_file = open(TARGET_FILE, "w+")
for i, batch in enumerate(test_iter):
    logging.info("Iteration {} started!".format(i))
    src = batch.src.transpose(0, 1)
    src = src.cuda()
    src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
    #logging.info("Source Size = {}".format(src.size()))
    out = greedy_decode(model, src, src_mask, max_len=600, start_symbol=TGT.vocab.stoi[BOS_WORD])
    for j in range(out.size(0)):
        #write_file.write('Translation:' + '\t')
        for k in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[j, k]]
            if sym == EOS_WORD:
                break
            write_file.write(sym + ' ')
        write_file.write('\n')
        #target_file.write('Target:' + '\t')
    
        for k in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[k, j]]
            if sym == EOS_WORD:
                break
            target_file.write(sym + ' ')
        target_file.write('\n')
target_file.close()
write_file.close()
