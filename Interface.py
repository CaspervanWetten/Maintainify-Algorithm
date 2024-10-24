import random
import logging
import os
import tqdm
import math
import numpy as np
import hashlib

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
    tokenizer:      str     # file_path naar bestand
    data                    = None
    train_data              = None
    test_data               = None
    val_data                = None
    model:          "Model" = None
    m_sate:         dict    = None
    debug_bool:     bool    = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Generic HP
    batch_size:     int     = 64    
    block_size:     int     = 16    # Equivalent to the sequence length (T in B,T,C)
    max_iters:      int     = 1200  
    embedding_dim:  int     = 384   
    patch_embedding: int    = 384 # Letterlijk het maximum totaal aantal tokens waar de trans context over onthoudt
    n_head:         int     = 6     
    n_layer:        int     = 6     
    dropout:        float   = 0.3   
    num_classes:    int     = 16 # The amount of different possible classifications; TODO make this data dependent and (probably) move it over to the tokenizer
    # Training HP
    eval_interval:  int     = 300   
    learning_rate:  float   = 1e-3  
    

    


    def __post_init__(self) -> None:
        self.logging = logging.getLogger(__name__); logging.basicConfig(level=logging.DEBUG if self.debug_bool else logging.WARNING)
        self.debug = lambda s: self.logging.debug(f"\n{s}\n")

        self.load_tokenizer(self.tokenizer)
        
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
    
    def load_tokenizer(self, file_path):
        # By far niet het mooiste, maar het werkt; op python 3.10+ had ik match-switch cases gebruikt
        # LET: kan een file_path zijn naar een bestaande, of een indicator voor text of audio
        if file_path.lower().endswith(".vocab"): # TODO Figure out either a secondary .name or a good way to differentiate
            self.Tok = TextTokenizer()
            self.Tok.load_tokenizer(file_path)
            return True # Early return, .vocabs don't need to be trained
        elif file_path.lower().endswith(".wav"):
            self.Tok = AudioTokenizer(debug = self.debug_bool)
        elif file_path.lower().endswith(".mp3"):
            self.Tok = AudioTokenizer(debug = self.debug_bool)
        elif file_path.lower().endswith(".txt"):
            self.Tok = TextTokenizer(debug = self.debug_bool)
            self.Tok.train(file_path)
        else:
            print("Tokenizer param should point to a either a data of .vocab file")
            raise NameError
        return True

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

    def optimize_categorization(self):
        # params = self.model.classification_head.parameters() + self.model.parameters()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        for iter in range(self.max_iters):
            if iter  == 1:
                print(f'Training loop started')
            
            self.model.train()
            # NOT the batching function
            # Im hardcoding tupled train_data (categorization)

            # This should be part of the model functionality
            for target in self.train_data:
                self.debug(f"training on {target}")
                batched_target = self.batch(target)
                for sample in tqdm.tqdm(self.train_data[target], total=len(self.train_data[target])):
                    self.debug(f"training iteration: {self.train_data[target].index(sample) + 1}")
                    batches = self.batch(sample)
                    out, loss = self.model(batches)
                    out, loss = self.model.classification_head(out, batched_target)
                    optimizer.zero_grad()
                    self.debug(f"out: {out}")
                    self.debug(f"loss: {loss}")
                    loss.backward()
                    optimizer.step()
                


            
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
        self.model.eval()
        correct = 0
        total = 0
        data_dict = {'train': self.train_data, 'val': self.val_data}
        with torch.no_grad():
            for split in ['train', 'val']:
                i = 0
                self.debug(f"split: {split}")
                for label, tensors in data_dict[split].items():
                    i+=1
                    batches = self.batch(torch.cat([tensor for tensor in tensors], dim=0))
                    torch.set_printoptions(profile='full')
                    self.debug(f"tensors iter: {i}\nwith tensors: {batches}")
                    torch.set_printoptions(profile='default')
                    embeddings = self.model.get_embeddings(batches)
                    pooled = embeddings.mean(dim=1)
                    logits = self.model.classification_head(pooled)
                    predictions = logits.argmax(dim=1)
                    correct += (predictions == label).sum().item()
                    self.debug(f"correct??? {correct}")

                    total += predictions.size(0)
        
        return correct / total if total > 0 else 0.0

    def load_data(self, path, split=None) -> None:
        # Could be more Pythonic (literally just ask GPT) but this is nice and explicit :)
        if split == "generate":
            return self.Tok.load_data(path)
        if split == "train":
            self.train_data = self.Tok.load_data(path)
        elif split == "val":
            self.val_data = self.Tok.load_data(path)
        elif split == "test":
            self.test_data = self.Tok.load_data(path)
        elif split == "return":
            return self.Tok.load_data(path)
        else:
            all = self.Tok.load_data(path)
            size = all.size(0)
            self.train_data, self.val_data, self.test_data = all[:int(size*0.7)], all[int(size*0.7):int(size*0.9)], all[int(size*0.9):] # een 70/20/10 split

    def load_categorized_data(self, path):
        """
        FORCES A TRAIN, VAL, TEST SPLIT
        """
        if not os.path.isdir(path):
            raise AttributeError("Passed path is not a folder!")

        self.train_data = {}
        self.test_data = {}
        self.val_data = {}
        for folder_name in os.listdir(path):           
            full_path = os.path.join(path, folder_name) # full_path here us passed_folder/category1
            if os.path.isdir(full_path): 
                data = []
                for file in (os.listdir(full_path)):
                    try:
                        data.append(self.Tok.encode(os.path.join(full_path, file)))
                    except Exception as e:
                        self.debug(f"Loading file {file} failed with error \n{e}")
                split1 = int(0.7 * len(data))  # 70% for training
                split2 = int(0.9 * len(data))  # 20% for validation, 10% for test
                train_data = data[:split1]
                val_data = data[split1:split2]
                test_data = data[split2:]
                folder_tensor = torch.tensor(self.generate_tensor_from_string(folder_name))
                self.debug(f"folder name: {folder_name}\ntensorized to: {folder_tensor}")
                self.train_data[folder_tensor] = train_data
                self.val_data[folder_tensor] = val_data
                self.test_data[folder_tensor] = test_data
            else:
                raise AttributeError("Passed folder does not have categorized data???")
        # self.debug(f"Finalized importing train, val and test data\n")
        # self.debug(f"train: {self.train_data}")
        # self.debug(f"val: {self.val_data}")
        # self.debug(f"test: {self.test_data}")
        # self.debug(f"self.train:{self.train_data}\nV normal train: {train_data}")
        # self.debug(f"split values: {split1}\n2: {split2}\nlen: {len(data)}")

    def generate_tensor_from_string(saelf, s, shape=(96, 96)):
        # Create a hash of the string
        hash_object = hashlib.md5(s.encode())
        hash_hex = hash_object.hexdigest()
        
        # Use the hash to seed the random number generator
        seed = int(hash_hex, 16) % (2**32 - 1)
        np.random.seed(seed)
        
        # Generate a random 3x3 tensor
        return np.random.rand(*shape)



    # Alias functions, shadow the model functionality alias interfacing
    def generate(self, input=None, max_new_tokens=4):
        """
        Alias for the model.categorize; also handles interfacing (i.e., translating input to a tensor if it isn't already)
        """
        encoded = self.Tok.encode(input)
        batches = self.batch(encoded)
        for _ in range(max_new_tokens):
            if batches.dtype != torch.long:
                print(f"ERROR \ncontext tensor should be of datatype Long, given tensor is of type: \n{batches.dtype}\n")
                return 
            self.debug(f"Currently generating token {_ + 1}")
            # crop context to the last block_size tokens
            cropped = batches[:, -self.block_size:]
            generated = self.model.generate(cropped)
            # context_wnew_token = torch.cat((input, self.model.generate(batch, self.block_size)), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
            batches = torch.cat((batches, generated), dim=1) # Reassign context to context_wnew_token to ensure the new tokens are taken into consideration for continous generation
        return batches

    def classify(self, input):
        """
        Alias for the model.categorize; also handles interfacing (i.e., translating categorize to a tensor if it isn't already)
        Classify an audio file and return prediction probabilities
        """
        # Claude
        self.model.eval()
        encoded = self.Tok.encode(input)
        batches = self.batch(encoded)
        with torch.no_grad():
            probs = self.model.classify(batches)
        class_probs = {f"class_{i}": prob.item() for i, prob in enumerate(probs[0])}
        return class_probs

    def pad(self, tensor_to_pad):
        padding_size = self.block_size - len(tensor_to_pad)
        padded_tensor = torch.cat((tensor_to_pad, torch.zeros(padding_size, dtype=tensor_to_pad.dtype)), dim=0)
        return padded_tensor

    def batch(self, context):
        if context.dim() == 1: # Als het een 1d tensor is, maak het een 2d tensor
            context = context.unsqueeze(0)
        batches = []
        # Claude:
        j = 0
        for row in context:
            j+=1
            for i in range(0, len(row), self.block_size):
                # if i % 1500 == 0:
                #     self.debug(f"batching iteration {i} of row {j}")
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
        stacked = torch.stack(batches)
        return stacked

        #Mijn:
        batches = []
        # # van 
        # for token in range(0, self.batch_size, to_batch // self.batch_size):
        #     batch_length = token + self.batch_size
        #     batch = to_batch[:, ]
        #     batch.append(batches)

    class Model(nn.Module):
        def __init__(self, Transformer: "Transformer") -> None:
            super().__init__()
            self.Transformer = Transformer
            self.vocab_size = Transformer.Tok.vocab_size

             # Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
            self.token_embedding_dimmingtable = nn.Embedding(self.vocab_size, self.Transformer.embedding_dim)

            """Note that the sequence they appear is also the sequence they are used"""

            #We're not just encoding identity, we're also encoding position!
            self.positioembedding_dimding_table = nn.Embedding(self.Transformer.block_size, self.Transformer.embedding_dim)
            self.blocks = nn.Sequential(*[self.Transformer.Block(self.Transformer, self.Transformer.embedding_dim, n_head=self.Transformer.n_head) for _ in range(self.Transformer.n_layer)])
            self.ln_f = nn.LayerNorm(self.Transformer.embedding_dim) # Final layer norm
            self.lm_head = nn.Linear(self.Transformer.embedding_dim, self.vocab_size) # LM=loaded model
            self.classification_head = nn.Linear(self.Transformer.embedding_dim, self.Transformer.num_classes)
            # N_embed is the number of embedded dimentions
            # .Embedding creates a shape of vocab_size x vocab_size
            # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table

        def generate(self, context: torch.Tensor, slice: int=1):
            """"
            Expects a long tensor
            Generates  óne (1) token
            """
                # raise AttributeError
            the_entire_tensor, expected_error = self(context) # Does the prediction 
            newest_math = the_entire_tensor[:, -slice, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
            prob_new_token = F.softmax(newest_math, dim=-1) # apply softmax to get probabilities, ook (B,C)
            new_token = torch.multinomial(prob_new_token, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
            return new_token

        def classify(self, input):
            if input.dtype != torch.long:
                print(f"WARNING \ncontext tensor should be of datatype Long, given tensor is of type: \n{input.dtype}\n Now converting using backup converting algorithm\n")
                input = float_to_long_tensor(input)
                
            embeddings = self.get_embeddings(input)
            pooled = embeddings.mean(dim=1)
            logits = self.classification_head(pooled)
            probs = F.softmax(logits, dim=-1)
            return probs

        def get_embeddings(self, tensor):
            # Claude
            B, T = tensor.shape
            tok_emb = self.token_embedding_dimmingtable(tensor)
            pos_emb = self.positioembedding_dimding_table(torch.arrange(T, device=self.Transformer.device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            return x


        def forward(self, context, targets=None):
            # Ik snap dit niet 
            """"
            Vereist een 2d tensor (voor nu)
            Moet voortaan vanuit de tokenizer komen 
            """
            B, T = context.shape
            # self.Transformer.debug(f"B: {B}\nT: {T}\ncontext max: {context.max()}\ncompared to vocab: {self.vocab_size}")
            # #context and targets are both (B,T) tensor of integers
            tok_em = self.token_embedding_dimmingtable(context)  # B,T,C 
                    # self.Tokenizer.embedding_
            pos_em = self.positioembedding_dimding_table(torch.arange(T, device=self.Transformer.device)) # T, C
            x = tok_em + pos_em # B,T,
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
                self.Transformer.debug(f"logits before: {logits}")
                logits = logits.view(B*T, C) # enkel 2d array, conformt aan cross_entropy functie, zelfde moet gebeurten bij targets
                self.Transformer.debug(f"logits after: {logits}")
                self.Transformer.debug(f"B: {B}\nT: {T}\nC: {C}")
                # Claude: Was enkel targets = targets.view(B * T)
                self.Transformer.debug(f"targets: {targets}")
                self.Transformer.debug(f"logits shape: {logits.shape}")
                self.Transformer.debug(f"targets shape: {targets.shape}")
                if targets.dim() == 2:
                    targets = targets.view(B * T)
                elif targets.dim() == 1 and targets.size(0) == B * T:
                    pass  # targets is already the correct shape
                else:
                    raise ValueError(f"Unexpected targets shape: {targets.shape}. Expected ({B}, {T}) or ({B * T},)")
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
            self.projection = nn.Linear(self.Transformer.patch_embedding, self.Transformer.embedding_dim) # patch embedding v encoder embedding
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