from torch import nn
import torch
from einops import rearrange
from typing import List

import math


class SelfAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_head, 
                 dropout_p) -> None:
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.dropout_p = dropout_p
        assert self.dim % num_head == 0


        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.drop_atten = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        # x: (b, l, d)
        # q, k, v: (b, l, dim_h)
        b, l, d = x.shape   
        q, k, v = torch.chunk(self.qkv_linear(x), chunks=3, dim=-1)

        # q, k, v: (b, num_head, l, dim_h)
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_head)
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_head)
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_head)

        # w: (b, num_head, l, l)
        w = q @ torch.transpose(k, -2, -1) / math.sqrt(self.dim / self.num_head)
        # / sqrt(dim_hidden) to make sure w.var() ~= 1

        if mask is not None:
            # mask of shape (b, l, l) -> (b, h, l, l)
            w = torch.masked_fill(w, mask.unsqueeze(1).to(x.device), float('-inf'))
            w = self.drop_atten(self.softmax(w))
        else:
            mask = ~torch.tril(torch.ones((l, l), device=x.device)).bool()
            w = torch.masked_fill(w, mask.to(x.device), float('-inf'))
            w = self.drop_atten(self.softmax(w))

        # w @ v: (b, num_head, l, l) @ (b, num_head, l, dim_h) = (b, num_head, l, dim_h) 
        # transpose --> (b, l, num_head, dim_h) 
        out = w @ v
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_linear(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_ffn=None) -> None:
        super().__init__()
        if dim_ffn is None:
            dim_ffn = dim_in * 4

        self.linear_in = nn.Linear(dim_in, dim_ffn)
        self.linear_out = nn.Linear(dim_ffn, dim_in)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear_out(self.act(self.linear_in(x)))
        return x



class TransformerBlock(nn.Module):
    def __init__(self, dim_in, num_head, dropout_p, dim_ffn=None) -> None:
        super().__init__()
        self.attention_layer = SelfAttention(dim_in, num_head, dropout_p)
        self.feed_forward = FeedForward(dim_in, dim_ffn)

        self.layer_norm1 = nn.LayerNorm(dim_in)
        self.layer_norm2 = nn.LayerNorm(dim_in)


        self.atten_drop = nn.Dropout(dropout_p)
        self.feed_forward_drop = nn.Dropout(dropout_p)


    def forward(self, x, mask):
        x = x + self.atten_drop(self.attention_layer(self.layer_norm1(x), mask))
        x = x + self.feed_forward_drop(self.feed_forward(self.layer_norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 vac_size: int, 
                 dim: int,
                 num_head: int,
                 dim_ffn: int,
                 dropout_p: float,
                 max_context_len: int,
                 ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vac_size, dim)
        self.pos_embed = nn.Embedding(max_context_len, dim)
        self.dropout_embed = nn.Dropout(dropout_p)
        
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_head, dropout_p, dim_ffn) 
                                     for _ in range(num_layers)])

        self.layer_norm_out = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, vac_size)



    def forward(self, x, mask=None):
        # b, l, d, l, d
        b, l = x.shape
        pos_encoding = self.pos_embed(torch.arange(l, device=x.device).unsqueeze(0))
        x = self.dropout_embed(self.embed(x) + pos_encoding)
        for layer in self.blocks:
            x = layer(x, mask)
        x = self.linear_out(self.layer_norm_out(x))
        return x
    
    
    @torch.no_grad()
    def get_target_from_src(self, x):
        trg = x.clone()
        trg[:, :-1] = x[:, 1:]
        trg[:, -1] = -100
        return trg
    
    def train_loss(self, inp_idxes):
        assert inp_idxes.ndim == 2
        # inp_idxes: (b, l)
        targets = self.get_target_from_src(inp_idxes) # (b, l)
        pred = self(inp_idxes) # (b, l, d)
        loss = torch.nn.functional.cross_entropy(pred.permute(0, 2, 1), targets)
        return loss

    def sample(self, 
               str_strings: List[str]=['春', '夏', '秋', '冬', '春日'],
               device = 'cuda') -> List[str]:
        sos_token_idx = self.word2idx['<SOS>']
        # eos_token_idx = self.word2idx['<EOS>']
        decode = lambda x: ''.join([self.idx2word[i] for i in x])
        encode = lambda x: [self.word2idx[w] for w in x]
        
        samples = []
        print('sampling...')
        for s in str_strings:
            inp = torch.tensor([sos_token_idx] + encode(s)).long().to(device)
            inp = inp.unsqueeze(0)
            for _ in range(50):
                out = self(inp)
                assert out.shape[0] == 1
                next_token_p = torch.nn.functional.softmax(out[0, -1:, :], dim=-1)
                
                # next_token_idx = torch.argmax(next_token_p, dim=1, keepdim=True)
                # do random sample
                next_token_idx = torch.multinomial(next_token_p, 1)
                inp = torch.cat([inp, next_token_idx], dim=1)
            dec_text = decode(inp.cpu()[0].tolist())
            print('\t', dec_text)
            samples.append(dec_text)
        return samples