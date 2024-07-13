import torch


def encode(x, word2idx):
    return [word2idx.get(w, -100) for w in x]

def decode(x: torch.Tensor, idx2word):
    return ''.join([idx2word.get(i, '-100') for i in x.tolist()])

def encode_batch(batch, word2idx):
    return [encode(x, word2idx) for x in batch]

def decode_batch(batch, idx2word):
    return [decode(x, idx2word) for x in batch]
