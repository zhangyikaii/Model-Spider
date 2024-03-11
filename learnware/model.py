import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from einops import repeat

from .learnware_info import MODEL2FEAT_DIM, BKB_SPECIFIC_RANK


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        log_attn = F.log_softmax(attn, 2)

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        output, attn, log_attn = self.attention(q, k, v, attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class LearnwareCAHeterogeneous(nn.Module):
    def __init__(self, *,
                 num_learnware: int,
                 dim: int,
                 hdim: int,
                 heads: int,
                 uni_hete_proto_dim: tuple,
                 data_sub_url: str,
                 pool: str = 'cls',
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 mode='concat-out-feature',
                 heterogeneous_extra_prompt=False):

        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert mode in {'concat-out-feature', 'concat-in-feature'}

        self.model_prompt = nn.Parameter(torch.randn(1, num_learnware, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = MultiHeadAttention(
            n_head=heads,
            d_model=hdim,
            d_k=hdim,
            d_v=hdim,
            dropout=dropout
        )
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

        uni_and_hete_dim = 1024
        self.uni_linear = nn.Linear(MODEL2FEAT_DIM[data_sub_url], uni_and_hete_dim)
        self.hete_linears = nn.ModuleDict({i_bkb: nn.Linear(MODEL2FEAT_DIM[i_bkb], uni_and_hete_dim) for i_bkb in BKB_SPECIFIC_RANK})

        self.uni_prompt = nn.Parameter(torch.randn(1, uni_hete_proto_dim[0], dim))
        self.hete_prompt = nn.Parameter(torch.randn(1, uni_hete_proto_dim[1], dim))

        self.hete_extra_prompt = heterogeneous_extra_prompt

    def forward(self, x_uni, x_hete, attn_mask, attn_mask_func=None, permute_indices=None):
        b = x_uni.shape[0]

        model_prompt = repeat(self.model_prompt, '1 c d -> b c d', b=b)

        uni_prompt = repeat(self.uni_prompt, '1 c d -> b c d', b=b)
        hete_prompt = repeat(self.hete_prompt, '1 c d -> b c d', b=b)

        outputs = []
        for i_prompt in range(model_prompt.shape[1]):
            cur_prompt = model_prompt[:, i_prompt, :].unsqueeze(1)
            if self.hete_extra_prompt:
                x_uni = x_uni + uni_prompt

            if attn_mask_func is not None:
                if self.hete_extra_prompt:
                    x_hete[i_prompt] = x_hete[i_prompt] + hete_prompt
                cur_x = torch.cat([cur_prompt, x_uni, x_hete[i_prompt]], dim=1)
                cur_attn_mask = attn_mask_func(attn_mask[i_prompt], 1)
            else:
                cur_x = torch.cat([cur_prompt, x_uni], dim=1)
                cur_attn_mask = attn_mask
            cur_x = self.transformer(cur_x, cur_x, cur_x, cur_attn_mask)
            cur_x = cur_x.mean(dim=1) if self.pool == 'mean' else cur_x[:, 0]
            cur_x = self.to_latent(cur_x)
            cur_x = self.mlp_head(cur_x)
            outputs.append(cur_x)

        outputs = torch.cat(outputs, dim=-1)

        return outputs
