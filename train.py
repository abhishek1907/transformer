import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets

from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch
from transformer.noam_opt import NoamOpt
import logging

LOG_FILE = './log/logged.log'
logging.basicConfig(filename= LOG_FILE, filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
# GPUs to use
devices = [0]  # Or use [0, 1] etc for multiple GPUs
BATCH_SIZE = 5000
SAVE_EVERY = 1
SAVE_DIR = './saved_models/fin_nlayer_3'
EPOCHS = 100
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'


def tokenize_de(text):
    return text.split()
    # return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return text.split()
    # return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

if True:
    # spacy_de = spacy.load('de')
    # spacy_en = spacy.load('en')

    data_fields = [('src', SRC), ('trg', TGT)]

    logging.info("Loading dataset from CSV")
    train, val, test = data.TabularDataset.splits(path='./data', train='train_fin.csv', validation='val_fin.csv', test='test_fin.csv', format='csv', fields=data_fields, skip_header=True)

    SRC.build_vocab(train.src)
    TGT.build_vocab(train.trg)


    logging.info("Vocab Loaded!")

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    model = make_model(len(SRC.vocab), len(TGT.vocab), n=3, dropout=0.3)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model.src_embed[0].d_model, 2, 4000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
    for epoch in range(EPOCHS):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))

        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
        logging.info("Loss at epoch {} = {}".format(epoch, loss))
        if epoch % SAVE_EVERY == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_par_state_dict': model_par.state_dict(),
            'loss': loss,
            'SRC' : SRC,
            'TGT' :TGT
            }, SAVE_DIR + 'model_{}_{}.pt'.format(epoch, loss))

