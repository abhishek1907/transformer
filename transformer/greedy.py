import torch
from torch.autograd import Variable

from .functional import subsequent_mask
import logging

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        #tgt_mask = torch.zeros(batch_size, ys.size(1), ys.size(1))    
        #tgt_mask[:] = subsequent_mask(ys.size(1))
        #out = model.decode(memory, src_mask, Variable(ys), Variable(tgt_mask).type_as(src.data))
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1))).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.view(batch_size, 1).type_as(src.data)], dim=1)
    return ys
