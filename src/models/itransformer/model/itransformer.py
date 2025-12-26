import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.embed import DataEmbedding_inverted
from ..layers.self_attention_family import AttentionLayer, FullAttention
from ..layers.transformer_enc_dec import Encoder, EncoderLayer


class ITransformer(nn.Module):
    """
    Modified ITransformer that outputs last hidden states instead of predictions.
    Based on: https://arxiv.org/abs/2310.06625
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        factor: int = 5,
        embed_type: str = "fixed", # not used
        freq: str = "h", # not used
        use_norm: bool = True,
        output_attention: bool = False,
        no_embd: bool = False,
    ):
        super(ITransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.no_embd = no_embd
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            seq_len, d_model, embed_type, freq, dropout
        )
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        # Optional projection head for downstream tasks
        self.projection = None

    def extract_features(self, x_enc, x_mark_enc):
        """Extract features/hidden states from input sequences."""
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
            # Store normalization stats for potential downstream use
            self._norm_stats = {'means': means, 'stdev': stdev}

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E (inverted from vanilla Transformer's B L E)
        if self.no_embd:
            enc_out = x_enc.permute(0, 2, 1)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # B N E -> B N E
        # Process inverted dimensions with attention, layernorm and ffn
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        return enc_out, attns

    def forward(self, x_enc, x_mark_enc=None, return_all_hiddens=False):
        """
        Args:
            x_enc: Input sequences [B, L, N]
            x_mark_enc: Time features [B, L, T] (optional)
            return_all_hiddens: If True, return all hidden states; 
                              if False, return only the last hidden state
        
        Returns:
            If output_attention is True:
                hidden_states [B, N, E] or [B, E], attention_weights
            Else:
                hidden_states [B, N, E] or [B, E]
        """
        # Extract features
        enc_out, attns = self.extract_features(x_enc, x_mark_enc)
        
        # enc_out shape: [B, N, E]
        hidden_states = enc_out
        
        if self.output_attention:
            return hidden_states, attns
        else:
            return hidden_states
