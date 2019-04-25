import logging
import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets
from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from transformer.my_iterator import MyIterator, rebatch
WRITE_FILE = './log/translations.txt'
LOG_FILE = './log/logged.log'
logging.basicConfig(filename= LOG_FILE, filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

SAVE_DIR = './saved_models/'
MODEL_FILE = 'model_10_1.9577244238077522.pt'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
BATCH_SIZE = 1000


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


save_dict = torch.load(SAVE_DIR+MODEL_FILE)
SRC = save_dict['SRC']
TGT = save_dict['TGT']
model = make_model(len(SRC.vocab), len(TGT.vocab), n=6)
model.load_state_dict(save_dict['model_state_dict'])
model.cuda()
logging.info('Model Loaded!')
data_fields = [('src', SRC), ('trg', TGT)]

train, val, test = data.TabularDataset.splits(path='./data', train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=data_fields, skip_header=True)

test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
logging.info('Started Decoding process!')
write_file = open(WRITE_FILE, "w+")
for i, batch in enumerate(test_iter):
    logging.info("Iteration {} started!".format(i))
    src = batch.src.transpose(0, 1)
    src = src.cuda()
    src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
    #logging.info("Source Size = {}".format(src.size()))
    out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=TGT.vocab.stoi[BOS_WORD])
    for j in range(out.size(0)):
        write_file.write('Translation:' + '\t')
        for k in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[j, k]]
            if sym == EOS_WORD:
                break
            write_file.write(sym + ' ')
        write_file.write('\n')
        write_file.write('Target:' + '\t')
    
        for k in range(batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[k, j]]
            if sym == EOS_WORD:
                break
            write_file.write(sym + ' ')
        write_file.write('\n')
write_file.close()
