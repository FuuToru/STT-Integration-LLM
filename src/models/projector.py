import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRippleAttention(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=5, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, seq_len, dim]
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [batch_size, seq_len, num_heads, head_dim]
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Local window-based attention
        local_attn_scores = torch.zeros(B, self.num_heads, N, N, device=x.device)
        for i in range(N):
            start = max(0, i - self.window_size // 2)
            end = min(N, i + self.window_size // 2 + 1)
            q_local = q[:, :, i:i+1, :]  # [batch_size, num_heads, 1, head_dim]
            k_local = k[:, :, start:end, :]  # [batch_size, num_heads, window_size, head_dim]
            v_local = v[:, :, start:end, :]  # [batch_size, num_heads, window_size, head_dim]
            attn = (q_local @ k_local.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, 1, window_size]
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            attn = attn.squeeze(2)  # [batch_size, num_heads, window_size]
            local_attn_scores[:, :, i, start:end] = attn

        # Global sparse attention (top-k)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        topk_attn, topk_indices = torch.topk(attn_scores, k=N//4, dim=-1)
        topk_attn = self.softmax(topk_attn)
        topk_attn = self.dropout(topk_attn)

        # Combine local and global attention
        local_output = (local_attn_scores @ v).transpose(1, 2).reshape(B, N, C)
        global_output = torch.zeros_like(v)
        for b in range(B):
            for h in range(self.num_heads):
                global_output[b, h] = torch.scatter(
                    global_output[b, h], dim=0, index=topk_indices[b, h], src=(topk_attn[b, h] @ v[b, h])
                )
        global_output = global_output.transpose(1, 2).reshape(B, N, C)
        output = (local_output + global_output) / 2.0

        # Final projection
        output = self.out(output)
        return output

class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim

        self.ripple_attn = SimpleRippleAttention(
            dim=self.encoder_dim,
            num_heads=getattr(config, 'num_heads', 2),
            window_size=getattr(config, 'window_size', 5),
            dropout=getattr(config, 'dropout', 0.1)
        )

        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        x = self.ripple_attn(x)  # [batch_size, seq_len, encoder_dim]

        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous().view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj
