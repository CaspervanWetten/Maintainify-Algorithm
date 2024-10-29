
import torch
import torch.nn as nn
from torch.nn import functional as F


"""
Relevant configs:

embedding_dim
block_size
dropout
n_heads
patch_embedding
"""

# TODO: Implementeer loss
# TODO: Kijk hier nog een keer naar om zeker te weten of je het snapt
# M TODO: Implementeer dit als @dataclass indien nodig blijkt te zijn
class TransformerModel(nn.Module):
    def __init__(self, config, logger) -> None:
        super().__init__()
        self.logger     = logger
        self.device     = config.device
        input_vocab     = config.input_vocab
        output_vocab    = config.output_vocab
        embedding_dim   = config.embedding_dim
        block_size      = config.block_size
        n_layers        = config.n_layers

            # Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
        self.token_embedding = nn.Embedding(input_vocab, embedding_dim)
        """De embedding is de 'lookup tabel' voor alle tokens. Als input verwacht het de totaal aantal
        tokens waar mee gerekend wordt, en het aantal dimensies wat elke token individueel krijgt"""
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        """Dit verschilt van de token embedding doordat het naar blokjes van tokens kijkt, ipv
        individuele tokens"""

        """Note that the sequence they appear is also the sequence they are used"""

        #We're not just encoding identity, we're also encoding position!
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim) # Final layer norm
        self.lm_head = nn.Linear(embedding_dim, output_vocab) # LM=loaded model
        # N_embed is the number of embedded dimentions
        # .Embedding creates a shape of vocab_size x vocab_size
        # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table

    def forward(self, context: torch.Tensor, targets=None):
        # Ik snap dit erg slecht
        B, T = context.shape
        tok_em = self.token_embedding(context) # B,T,C
        pos_em = self.position_embedding(torch.arange(T, device=self.device))
        x = tok_em + pos_em # B,T
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B,T, output_vocab_size
        # Dit zijn eig. nog niet de logits maar de token embeddings, 
        # Voor daadwerkelijk werken moet dit nog een softmax krijgen TODO: Doe dit in de interface
        # Creeert B,T,C als voorspelde token
        # Still have to evaluate loss:
        if targets is None:
            loss = None
        else:
            # TODO: Pass dit aan
            pass
        return logits, loss


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim   = config.embedding_dim
        block_size      = config.block_size
        dropout         = config.dropout
        n_heads         = config.n_heads
        head_size = embedding_dim // n_heads

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self,  config):
        super().__init__()
        n_heads         = config.n_heads
        patch_embedding = config.patch_embedding
        embedding_dim   = config.embedding_dim
        dropout         = config.dropout

        self.heads = nn.ModuleList([Head(config) for _ in range(n_heads)])
        self.projection = nn.Linear(patch_embedding, embedding_dim) # patch embedding v encoder embedding
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out)) # A linear transformation of the output of the concationation
        return out
class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self,  config):
        super().__init__()
        embedding_dim   = config.embedding_dim
        dropout         = config.dropout

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embedding_dim = config.embedding_dim
        
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward()
        # This will be a slight deviation from "attention is all you need"
        # We will be doing pre-attention layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

