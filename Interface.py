import random
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

# from helpers import get_input
from time import sleep
from Tokenizers import AudioTokenizer, TextTokenizer
from dataclasses import dataclass
from helpers import *

@dataclass
class Transformer():
    # Other attribute
    tokenizer:      str     # Kan generiek "audio" of "text" (foto video ooit?) zijn, of specifiek een pad naar een .pt bestand
    data                    = None
    train_data: torch.Tensor= None
    test_data: torch.Tensor = None
    val_data:  torch.Tensor = None
    model:          "Model" = None
    m_sate:         dict    = None
    debug:          bool    = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Generic HP
    batch_size:     int     = 64    
    block_size:     int     = 16    
    max_iters:      int     = 1200  
    embedding_dim:  int     = 384   
    patch_embedding: int    = 384 # Letterlijk het maximum totaal aantal tokens waar de trans context over onthoudt
    n_head:         int     = 6     
    n_layer:        int     = 6     
    dropout:        float   = 0.3   
    # Training HP
    eval_interval:  int     = 300   
    learning_rate:  float   = 1e-3  
    

    


    def __post_init__(self) -> None:
        self.logging = logging.getLogger(__name__); logging.basicConfig(level=logging.DEBUG if self.debug else logging.WARNING)
        self.debug = lambda s: self.logging.debug(f"\n{s}\n")

        # By far niet het mooiste, maar het werkt; op python 3.10+ had ik match-switch cases gebruikt
        # LET: kan een file_path zijn naar een bestaande, of een indicator voor text of audio
        if self.tokenizer.lower() == "text":
            self.Tok = TextTokenizer()
        elif self.tokenizer.lower().endswith(".vocab"):
            self.Tok = TextTokenizer().load_model(self.tokenizer)
        elif self.tokenizer.lower() == "audio":
            self.Tok = AudioTokenizer(debug = False)
        elif self.tokenizer.lower().endswith(".aud"):
            self.Tok = AudioTokenizer().load_model(self.tokenizer)
        else:
            print("Tokenizer should either be a path to a pre-saved tokenizer, or a string indicating 'text' or 'audio'")
            raise NameError
        
        # Model initialization: TODO make this better lol
        self.model = self.Model(self)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        pass

    def save_model(self, file_path) -> bool:
        if saved_corectly:
            return True
        return False
    
    def load_model(self, file_path: str):
        if not file_path.lower().endswith(".pt") and not file_path.lower().endswith(".pth") and not file_path.lower().endswith(".model"):
            print("Models should be either '.pt', '.pth' or '.model' files")
            raise NameError
        try:
            self.model = torch.load(file_path)
        except Exception as e:
            print(f"Failed to load model with error\n{e}\nFalling back to untrained model")
            self.model = self.Model()
    
    def optimize(self) -> None:
        self.debug("Starting otimization")
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(losses)
                print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

            train_batches = self.batch(self.train_data)
            for batch in train_batches:
                logits, loss = self.model(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    def optimize_categorize(self):
        self.debug("Start categorization optimizaiton")
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0:
                metrics = self.estimate_categorization_loss()
                self.debug(f"step: {iter}")
                self.debug(f"train loss: {metrics['train_loss']:.4f}, accuracy {metrics['train_accuracy']:.4f}")
                self.debug(f"train loss: {metrics['val_loss']:.4f}, accuracy {metrics['val_accuracy']:.4f}")
            labels = list(self.train_data.keys())
            label = random.choice(labels)
            tensors = self.train_data[label]
            train_batches = self.batch(tensors)
            for batch in train_batches:
                logits, loss = self.model(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()


    @torch.no_grad() # A context manager (?) to tell PyTorch to not make these backwards callable, e.g. skip back propagation
    def estimate_loss(self):
        out = {}
        self.model.eval() # Set the model to eval mode
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.model._get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train() # Set the model to training mode
        return out

    @torch.no_grad()
    def estimate_categorization_loss(self):
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_interval)
            accuracies = torch.zeros(self.eval_interval)
            data_dict = self.train_data if split == "train" else self.val_data
            iter_count = 0

            for label, tensors in data_dict.items():
                # Figure out a way for the label to also be a tensor
                batches = self.batch(tensors)
                for batch in batches:
                    logits, loss = self.model(batch, label)
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == label).float().mean()
                    losses[iter_count] = loss.item()
                    accuracies[iter_count] = accuracy.item()
                    iter_count += 1

                    if iter_count > self.eval_interval:
                        break
                out[f'{split}_loss'] = losses.mean()
                out[f'{split}_accuracy'] = accuracies.mean()
        self.model.train()
        return out

    def load_data(self, path, split=None) -> None:
        possible = [None, "train", "val", "test", "generate"] 
        if split not in possible:
            print(f"split is not one of {possible}")
            raise NameError
        # Could be more Pythonic (literally just ask GPT) but this is nice and explicit :)
        if split == "generate":
            return self.Tok.load_data(path)
        if split == "train":
            self.train_data = self.Tok.load_data(path)
        elif split == "val":
            self.val_data = self.Tok.load_data(path)
        elif split == "test":
            self.test_data = self.Tok.load_data(path)
        else:
            all = self.Tok.load_data(path)
            size = all.size(0)
            self.train_data, self.val_data, self.test_data = all[:int(size*0.7)], all[int(size*0.7):int(size*0.9)], all[int(size*0.9):] # een 70/20/10 split



    # Alias functions, shadow the model functionality alias interfacing
    def generate(self, input=None, max_new_tokens=64):
        """
        Alias for the model.categorize; also handles interfacing (i.e., translating input to a tensor if it isn't already)
        """
        # if input == None:
        #         input = torch.zeros((9, 9), dtype=torch.long)
        # if not isinstance(input, torch.Tensor):
        #         input = self.load_data(input, "generate")
        #         self.debug(f"encoding input")
        # self.debug(f"encoded Input: {len(input)} ; {input}")
            
        # self.debug(f"input size(0): {input.size(0)}")
        # if input.size(1) > self.block_size:
        #     self.debug(f"batching input")
        #     input = self.batch(input)
        text = get_input("Wouter")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stringtoint = { ch:i for i,ch in enumerate(chars) } # {"A":0, "B":1, ..., "!": 80}
        inttostring= { i:ch for i,ch in enumerate(chars) } # {0:"A", 1:"B", ..., 80:"!"}
        self._bencode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
        self._bdecode = lambda l: ''.join([inttostring[i] for i in l]) # backup decoding algo
        self.distinct_tokens = chars
        self.num_distinct_tokens = vocab_size

        encoded = torch.tensor(self._bencode(input), dtype=torch.long)
        batches = self.batch(encoded)
        for _ in range(max_new_tokens):
            self.debug(f"Currently generating token {_ + 1}")
            # crop context to the last block_size tokens
            # cropped_context = batch[:, -self.block_size:]
            generated = self.model.generate(batches)
            # context_wnew_token = torch.cat((input, self.model.generate(batch, self.block_size)), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
            batches = torch.cat((batches, generated), dim=1) # Reassign context to context_wnew_token to ensure the new tokens are taken into consideration for continous generation
        return batches

    def categorize(self, input):
        """
        Alias for the model.categorize; also handles interfacing (i.e., translating categorize to a tensor if it isn't already)
        """
        return self.model.categorize(input)

    def pad(self, tensor_to_pad):
        self.debug(f"padding: \n{tensor_to_pad}")
        self.debug(f"Lengte van: \n{len(tensor_to_pad)}")
        padding_size = self.block_size - len(tensor_to_pad)
        self.debug(f"padding_size: \n{padding_size}")
        padded_tensor = torch.cat((tensor_to_pad, torch.zeros(padding_size, dtype=tensor_to_pad.dtype)), dim=0)
        self.debug(f"the padded tensor: {padded_tensor}")
        return padded_tensor

    def batch(self, context):
        if context.dim() == 1: # Als het een 1d tensor is, maak het een 2d tensor
            context = context.unsqueeze(0)
        batches = []
        self.debug(f"batching: {context}")
        # Claude:
        for row in context:
            for i in range(0, len(row), self.block_size):
                self.debug(batches)
                end = i + self.block_size
                if end > len(row):
                    batch = row[i:]
                    batches.append(self.pad(batch))
                else:
                    batch = row[i:end]
                    batches.append(batch)
        # for start in range(0, context.size(0) - self.block_size + 1, self.block_size // 2):
        #     end = start + self.block_size
        #     batch = context[:, start:end]
        #     batches.append(batch)
        self.debug(batches)
        return torch.stack(batches)

        #Mijn:
        batches = []
        # # van 
        # for token in range(0, self.batch_size, to_batch // self.batch_size):
        #     batch_length = token + self.batch_size
        #     batch = to_batch[:, ]
        #     batch.append(batches)
        
    def load_data(self, path):
        """
        Shadows tokenizer load data
        """



    class Model(nn.Module):
        def __init__(self, Transformer: "Transformer") -> None:
            super().__init__()
            self.Transformer = Transformer

            text = get_input("Wouter")
            chars = sorted(list(set(text)))
            vocab_size = len(chars)
            stringtoint = { ch:i for i,ch in enumerate(chars) } # {"A":0, "B":1, ..., "!": 80}
            inttostring= { i:ch for i,ch in enumerate(chars) } # {0:"A", 1:"B", ..., 80:"!"}
            self._bencode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
            self._bdecode = lambda l: ''.join([inttostring[i] for i in l]) # backup decoding algo
            self.distinct_tokens = chars
            self.num_distinct_tokens = vocab_size

             # Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
            self.token_embedding_dimmingtable = nn.Embedding(self.num_distinct_tokens, self.Transformer.embedding_dim)

            """Note that the sequence they appear is also the sequence they are used"""

            #We're not just encoding identity, we're also encoding position!
            self.positioembedding_dimding_table = nn.Embedding(self.Transformer.block_size, self.Transformer.embedding_dim)
            self.blocks = nn.Sequential(*[self.Transformer.Block(self.Transformer, self.Transformer.embedding_dim, n_head=self.Transformer.n_head) for _ in range(self.Transformer.n_layer)])
            self.ln_f = nn.LayerNorm(self.Transformer.embedding_dim) # Final layer norm
            self.lm_head = nn.Linear(self.Transformer.embedding_dim, self.num_distinct_tokens) # LM=loaded model
            # N_embed is the number of embedded dimentions
            # .Embedding creates a shape of vocab_size x vocab_size
            # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table

        def generate(self, context: torch.Tensor, slice: int=1):
            """"
            Expects a long tensor
            Generates  óne (1) token
            """
            if context.dtype != torch.long:
                raise AttributeError
            the_entire_tensor, expected_error = self(context) # Does the prediction 
            newest_math = the_entire_tensor[:, -slice, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
            prob_new_token = F.softmax(newest_math, dim=-1) # apply softmax to get probabilities, ook (B,C)
            new_token = torch.multinomial(prob_new_token, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
            return new_token

        def forward(self, context, targets=None):
            # Ik snap dit niet 
            """"
            Vereist een 2d tensor (voor nu)
            Moet voortaan vanuit de tokenizer komen 
            """
            B, T = context.shape
            self.Transformer.debug(B)
            self.Transformer.debug(T)
            #context and targets are both (B,T) tensor of integers
            tok_em = self.token_embedding_dimmingtable(context)  # B,T,C 
                    # self.Tokenizer.embedding_
            pos_em = self.positioembedding_dimding_table(torch.arange(T, device=self.Transformer.device)) # T, C
            x = tok_em + pos_em # B,T,
            # ----------------------------
            # Volgens mij moet alles hierboven de tokenizer worden
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x) # (B,T, vocab_size)
            # Not Logits, token embeddings
            
            # Creates a (Batch X Time X Channel) tensor
            # De 'logit' is dus de voorspelde volgende token (i.e. woord/whatever) zonder enige context.
            # Je wilt de loss nog evalueren
            
            if targets is None:
                loss = None
            else:
                # Dit werkt niet, de input van cross_entropy moet BxCxT zijn ipv BxTxC
                B, T, C = logits.shape
                logits = logits.view(B*T, C) # enkel 2d array, conformt aan cross_entropy functie, zelfde moet gebeurten bij targets
                targets = targets.view(B*T) # Voor reden die ik niet snap is targets.view(-1) hetzelfde
                loss = F.cross_entropy(logits, targets) # Hoe goed voorspel je het volgende karakter gebasseerd op de logit?
                # De waarde van 
                
                # Loss verwacht je dat -ln(1/vocab_size) is (ln=natuurlijk logarithme)
                
            return logits, loss

    class Head(nn.Module):
        def __init__(self, Transformer: 'Transformer', head_size):
            super().__init__()
            self.Transformer = Transformer
            self.key = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.query = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.value = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(self.Transformer.block_size, self.Transformer.block_size)))
            self.dropout = nn.Dropout(self.Transformer.dropout)
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
        def __init__(self,  Transformer: 'Transformer', num_heads, head_size):
            super().__init__()
            self.Transformer = Transformer
            self.heads = nn.ModuleList([self.Transformer.Head(self.Transformer, head_size) for _ in range(num_heads)])
            self.projection = nn.Linear(self.Transformer.embedding_dim, self.Transformer.embedding_dim) # patch embedding v encoder embedding
            self.dropout = nn.Dropout(self.Transformer.dropout)
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.projection(out)) # A linear transformation of the output of the concationation
            return out
    class FeedFoward(nn.Module):
        """A simple linear layer followed by a non-linearity"""
        def __init__(self,  Transformer: 'Transformer', embedding_dim):
            super().__init__()
            self.Transformer = Transformer
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
                nn.Dropout(self.Transformer.dropout)
            )
        def forward(self, x):
            return self.net(x)
    class Block(nn.Module):
        def __init__(self, Transformer: 'Transformer', embedding_dim, n_head) -> None:
            super().__init__()
            self.Transformer = Transformer
            head_size = embedding_dim // n_head
            self.sa = self.Transformer.MultiHeadAttention(self.Transformer, n_head, head_size)
            self.ffwd = self.Transformer.FeedFoward(self.Transformer, embedding_dim)
            # This will be a slight deviation from "attention is all you need"
            # We will be doing pre-attention layer normalization
            self.ln1 = nn.LayerNorm(embedding_dim)
            self.ln2 = nn.LayerNorm(embedding_dim)
        
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x 



if __name__ == "__main__": 
    # batches = []
    # for start in range(0, sequence_length - self.block_size + 1, self.block_size // 2):
    #     end = start + self.block_size
    #     batch = context[:, start:end]
    #     batches.append(batch)
    
    # For testing purposes
    pass