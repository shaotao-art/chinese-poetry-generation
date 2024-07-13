import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from functools import partial


def read_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

class PoeData(Dataset):
    def __init__(self, 
                 data_pkl_p: str,
                 word2idx_pkl_p: str, 
                 idx2word_pkl_p: str) -> None:
        super().__init__()
        self.pretokenized_data = read_pkl(data_pkl_p)
        self.word2idx = read_pkl(word2idx_pkl_p)
        self.idx2word = read_pkl(idx2word_pkl_p)
        
        
    def __getitem__(self, idx):
        # add start of sentence and end of sentence token
        return torch.tensor([self.word2idx['<SOS>']] + 
                            self.pretokenized_data[idx] + 
                            [self.word2idx['<EOS>']]).long()


    def __len__(self):
        return len(self.pretokenized_data)

def lm_attention_mask(seq_l: torch.Tensor):
    """return attention mask of (l, l)
    mask = True, mean it will be masked"""
    up_tril_mask = torch.tril(torch.ones((seq_l, seq_l)).bool())
    return ~up_tril_mask


def get_pad_mask(x: torch.Tensor, 
                 pad_token_idx: int):
    """construct mask token for batch data of (b, l, l)
    mask = True, mean it will be masked"""
    assert len(x.shape) == 2
    b, l = x.shape
    is_pad_token_mask = x == pad_token_idx # (b, l)
    is_pad_token_mask = is_pad_token_mask.unsqueeze(1) # (b, 1, l)
    return is_pad_token_mask

def collate_fn(batch, pad_token_idx, max_len):
    inps = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)

    # trunction 
    b_s, l = inps.shape
    if l > max_len:
        inps = inps[..., :max_len]
        # l = max_len
    # NOTE: do not need pad mask and attn mask
    # construct mask
    # attn_mask = lm_attention_mask(l) # (l, l)
    # pad_mask = get_pad_mask(inps, pad_token_idx) # (b, 1, l)
    # mask = torch.logical_or(attn_mask, pad_mask) # (b, l, l)
    return dict(inp=inps, 
                # mask=mask
                )
    
def get_dataset_loader(data_config):
    fn = partial(collate_fn, pad_token_idx=data_config.pad_token_idx, max_len=data_config.max_len)
    dataset = PoeData(**data_config.dataset_config)
    data_loader = DataLoader(dataset, collate_fn=fn, **data_config.data_loader_config)
    return dataset, data_loader