import torch
from torch import nn
from einops import rearrange
from torch import einsum
import math
from typing import List


"""
a modern transformer model with 
1. RoPE
2. RMSNorm
3. Group Query Attention
4. GLU
5. K, V cache
"""

class RoPe(nn.Module):
    def __init__(self, max_len, head_dim, base=10000):
        super(RoPe, self).__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        

    def _init_cosine_sine(self):
        pos_idx = torch.arange(0,self.max_len)
        freq = self.base ** - (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        cache = einsum('i, j -> i j', pos_idx, freq) # (max_len, dim // 2) 
        cos = torch.cos(cache)
        sin = torch.sin(cache)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x: torch.Tensor, start_idx = None):
        if not hasattr(self, 'cos'):
            self._init_cosine_sine()
            self.cos = self.cos.to(x.device)
            self.sin = self.sin.to(x.device)
            
        if start_idx is None:
            assert x.shape[-2] < self.max_len
            seq_len = x.shape[-2]
            even = x[..., ::2]
            odd = x[..., 1::2]
            even_res = even * self.cos[:seq_len] - odd * self.sin[:seq_len]
            odd_res = odd * self.cos[:seq_len] + even * self.sin[:seq_len]
            x[..., ::2] = even_res
            x[..., 1::2] = odd_res
            return x
        else:
            seq_len = x.shape[-2]
            # when in infer mode, model's input using kv cache is
            # (b, init_len, d)
            # or (b, 1, d) 
            # we should only apply rope's correct postional accord to start_idx
            assert start_idx < self.max_len
            even = x[..., ::2]
            odd = x[..., 1::2]
            even_res = even * self.cos[start_idx: start_idx + seq_len] - odd * self.sin[start_idx: start_idx + seq_len]
            odd_res = odd * self.cos[start_idx: start_idx + seq_len] + even * self.sin[start_idx: start_idx + seq_len]
            x[..., ::2] = even_res
            x[..., 1::2] = odd_res
            return x
        
class GQA(nn.Module):
    def __init__(self, num_head, head_dim, num_kv_groups, dropout, rope: RoPe):
        super(GQA, self).__init__()
        assert num_head % num_kv_groups == 0, 'num_head should be divisible by num_kv_groups'
        self.num_head = num_head
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.num_head_kv = num_head // num_kv_groups
        
        self.q_dim = num_head * head_dim
        self.kv_dim = self.num_head_kv * head_dim
        
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.q_dim, self.q_dim)
        self.k_proj = nn.Linear(self.q_dim, self.kv_dim)
        self.v_proj = nn.Linear(self.q_dim, self.kv_dim)
        
        self.rope = rope
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.q_dim, self.q_dim)
        

        self.k_cache = None
        self.v_cache = None
    
    def infer_forward(self, x, start_idx):
        # x shape: (batch, 1, q_dim)
        q = self.q_proj(x) # shape: (batch, 1, q_dim)
        k = self.k_proj(x) # shape: (batch, 1, kv_dim)
        v = self.v_proj(x) # shape: (batch, 1, kv_dim)
       
        # seq_len = q.shape[1]
        # mask = ~torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
         
        q = rearrange(q, 'b n (g h d) -> b g h n d', 
                      h=self.num_head_kv, 
                      g=self.num_kv_groups)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_head_kv) # (b, h, 1, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_head_kv) # (b, h, 1, d)

        # apply rope
        q = self.rope(q, start_idx=start_idx) # (b, g, h, 1, d)
        k = self.rope(k, start_idx=start_idx) # (b, h, 1, d)
        
        if start_idx == 0:
            self.k_cache = None
            self.v_cache = None
            
        if self.k_cache is not None:
            # cat cache
            k = torch.cat([self.k_cache, k], dim=-2) # (b, h, len(cache) + 1, d)
            v = torch.cat([self.v_cache, v], dim=-2) # (b, h, len(cache) + 1, d)
            # update cache
            self.k_cache = k
            self.v_cache = v
        else:
            # init k, v cache
            self.k_cache = k
            self.v_cache = v

        # print('>>>>>>')
        # print('q, k shape: ', q.shape, k.shape)
        # print('k, v cache shape: ', self.k_cache.shape, self.v_cache.shape)
        attn = einsum('b g h i d, b h j d -> b g h i j', q * self.scale, k) 
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # print('attn shape: ', attn.shape)
        
        out = einsum('b g h i j, b h j d -> b g h i d', attn, v)
        out = rearrange(out, 'b g h n d -> b n (g h d)')
        
        out = self.out_proj(out)
        return out
        
        

    def train_forward(self, x):
        # x shape: (batch, seq_len, q_dim)
        q = self.q_proj(x) # shape: (batch, seq_len, q_dim) 
        k = self.k_proj(x) # shape: (batch, seq_len, kv_dim)
        v = self.v_proj(x) # shape: (batch, seq_len, kv_dim)
        
        seq_len = q.shape[1]
        mask = ~torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        
        q = rearrange(q, 'b n (g h d) -> b g h n d', 
                      h=self.num_head_kv, 
                      g=self.num_kv_groups)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_head_kv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_head_kv)

        # apply rope
        q = self.rope(q)
        k = self.rope(k)
        
        attn = einsum('b g h i d, b h j d -> b g h i j', q * self.scale, k) 
        attn = torch.masked_fill(attn, mask=mask, value=float('-inf'))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = einsum('b g h i j, b h j d -> b g h i d', attn, v)
        out = rearrange(out, 'b g h n d -> b n (g h d)')
        
        out = self.out_proj(out)
        return out
    
    def forward(self, x, mask=None, start_idx=None):
        if start_idx is None:
            return self.train_forward(x)
        else:
            return self.infer_forward(x, start_idx=start_idx)

class GLU(nn.Module):
    def __init__(self, dim, dim_ffn = None):
        super().__init__()
        if dim_ffn is None:
            dim_ffn = dim * 4
        self.linear1 = nn.Linear(dim, dim_ffn)
        self.linear2 = nn.Linear(dim_ffn, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.affine = nn.Parameter(torch.ones((1, dim)))
        self.sqrt_d = math.sqrt(dim)
        
    def forward(self, x):
        # x: (b, l, d)
        return torch.nn.functional.normalize(x, p=2.0, dim=-1) * self.sqrt_d * self.affine

class TransformerLayer(nn.Module):
    def __init__(self, num_head, head_dim, dim_ffn, num_kv_groups, dropout, rope: RoPe):
        super(TransformerLayer, self).__init__()
        self.norm1 = RMSNorm(head_dim * num_head)
        self.gqa = GQA(num_head, head_dim, num_kv_groups, dropout, rope)
        self.drop_1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(head_dim * num_head)
        self.ffn = GLU(head_dim * num_head, dim_ffn=dim_ffn)
        self.drop_2 = nn.Dropout(dropout)
        

    def forward(self, x, start_idx=None):
        x = x + self.drop_1(self.gqa(self.norm1(x), start_idx=start_idx))
        x = x + self.drop_2(self.ffn(self.norm2(x)))
        return x
    
class GPT(nn.Module):
    def __init__(self, num_head, max_len, head_dim, num_kv_groups, dim_ffn, dropout, num_layers, num_classes):
        super(GPT, self).__init__()
        self.num_head = num_head
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.num_class = num_classes
        
        self.rope = RoPe(max_len, head_dim)
        
        self.wte = nn.Embedding(num_classes, head_dim * num_head)
        self.wte_dropout = nn.Dropout(dropout)
        
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(num_head, head_dim, dim_ffn, num_kv_groups, dropout, self.rope) 
             for _ in range(num_layers)])
        

        self.head = nn.Sequential(
            RMSNorm(head_dim * num_head),
            nn.Linear(head_dim * num_head, num_classes)
        )
        
    def forward(self, x, start_idx=None):
        x = self.wte_dropout(self.wte(x))
        
        for layer in self.transformer_layers:
            x = layer(x, start_idx=start_idx)
        x = self.head(x)
        return x # (b, l, num_classes)
    
    @torch.no_grad()
    def infer(self, init_inp):
        self.eval()
        # init_inp shape: (b, l)
        out = [init_inp]
        init_len = init_inp.shape[-1]

        x = init_inp
        start_idx = 0
        for it in range(50):
            pred = self(x, start_idx=start_idx) # (1, seq_len, num_classes)
            pred = pred[:, -1, :] # (1, num_classes)
            prob = torch.nn.functional.softmax(pred, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1) # (1, 1)
            x = next_token
            if start_idx == 0:
                start_idx += init_len
            else:
                start_idx += 1
            out.append(next_token)
        return torch.cat(out, dim=-1)
    
    @torch.no_grad()
    def get_target_from_src(self, x):
        trg = x.clone() # x (b, l)
        trg[:, :-1] = x[:, 1:]
        trg[:, -1] = -100
        trg[trg == self.word2idx['<PAD>']] = -100
        return trg
    
    def train_loss(self, inp_idxes):
        assert inp_idxes.ndim == 2
        # inp_idxes: (b, l)
        pred = self(inp_idxes) # (b, l, d)
        targets = self.get_target_from_src(inp_idxes)
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
            inp = torch.tensor([sos_token_idx] + encode(s)).long().to(device) # (b, init)
            inp = inp.unsqueeze(0)
            for _ in range(50):
                out = self(inp) # (b, l, num_class)
                assert out.shape[0] == 1
                next_token_p = torch.nn.functional.softmax(out[:, -1, :], dim=-1) # (b, num_class)
                next_token_idx = torch.multinomial(next_token_p, 1) # (b, 1)
                inp = torch.cat([inp, next_token_idx], dim=-1)
            dec_text = decode(inp.cpu()[0].tolist())
            print('\t', dec_text)
            samples.append(dec_text)
        return samples