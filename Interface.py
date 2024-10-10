import random
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

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
    block_size:     int     = 16    
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
        else:
            print("Tokenizer param should point to a either a data of .vocab file")
            raise NameError
        self.Tok.train(file_path)
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
        self.debug("Start categorization optimizaiton")
        best_val_acc = 0.0
        for iter in range(self.max_iters):
            self.model.train()
            total_loss = 0
            num_batches = 0
            if iter % 100 == 0:
                losses = self.estimate_categorization_loss()
                print(f"losses{losses}")
                print(f"avg_loss: {avg_loss}\nloss: {loss}")


            for label, tensors in self.train_data.items():
                label_tensor = torch.tensor([label])
                batches = self.batch(tensors)
                for batch in batches:
                    optimizer.zero_grad()
                    embeddings = self.model.get_embeddings()
                    pooled = embeddings.mean(dim=1)
                    logits = self.model.classification_head(pooled)
                    loss = F.cross_entropy(logits, label_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / num_batches

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
                    self.debug(f"tensors iter: {i}\nwith tensors: {tensors}")
                    j = 0
                    batches = self.batch(torch.cat([tensor for tensor in tensors], dim=1))
                    for batch in batches:
                        self.debug(f"batch j {j}")
                        embeddings = self.model.get_embeddings(batch)
                        pooled = embeddings.mean(dim=1)
                        logits = self.model.classification_head(pooled)
                        predictions = logits.argmax(dim=1)
                        correct += (predictions == label).sum().item()
                        self.debug(f"correct??? {correct}")

                        total += predictions.size(0)
        
        return correct / total if total > 0 else 0.0

    def load_data(self, path, split=None) -> None:
        possible = [None, "train", "val", "test", "generate",  "return"] 
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
            full_path = os.path.join(path, folder_name)
            if os.path.isdir(full_path):
                data = self.load_data(full_path, split="return")
                split1 = int(0.7 * len(data))  # 70% for training
                split2 = int(0.9 * len(data))  # 20% for validation, 10% for test
                train_data = data[:split1]
                val_data = data[split1:split2]
                test_data = data[split2:]
                self.train_data[folder_name] = train_data
                self.val_data[folder_name] = val_data
                self.test_data[folder_name] = test_data
            else:
                raise AttributeError("Passed folder does not have categorized")



    # Alias functions, shadow the model functionality alias interfacing
    def generate(self, input=None, max_new_tokens=4):
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
        # encoded = torch.tensor(self._bencode(input), dtype=torch.long)
        encoded = self.Tok.encode(input)
        batches = self.batch(encoded)
        for _ in range(max_new_tokens):
            if batches.dtype != torch.long:
                print(f"WARNING \ncontext tensor should be of datatype Long, given tensor is of type: \n{batches.dtype}\n Now converting using backup converting algorithm\n")
                batches = float_to_long_tensor(batches)

            torch.set_printoptions(profile='full')
            self.debug(batches)
            torch.set_printoptions(profile='default')


            self.debug(f"Currently generating token {_ + 1}")
            # crop context to the last block_size tokens
            cropped = batches[:, -self.block_size:]
            self.debug(f"batches: {batches}\n in contrast to: {cropped}")
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
        self.debug(f"batching: {context}")
        # Claude:
        for row in context:
            for i in range(0, len(row), self.block_size):
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
        self.debug(f"Batchified: {stacked}")
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
            self.Transformer.debug(f"input max: {tensor.max()}\n compared to vocab_size: {self.vocab_size}")
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
            self.Transformer.debug(f"B: {B}\nT: {T}\ncontext max: {context.max()}\ncompared to vocab: {self.vocab_size}")
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