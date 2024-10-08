import torch
import logging
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as ta_kaldi

import librosa
import os as os
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict




@dataclass
class GenericTokenizer(ABC):
    # Other attribute
    debug:        bool      = False
    vocab:          dict    = field(default_factory=dict)
    custom_tokens:  dict    = field(default_factory=dict)
    n_merges: int           =   512 # The amount of merges the default BPE algo uses
    merges: dict            = field(default_factory=dict) # The dictionary to keep track of which token x relates to the merged tuple (a,b)

    def __post_init__(self) -> None:
        self.logging = logging.getLogger(__name__); logging.basicConfig(level=logging.DEBUG if self.debug else logging.WARNING)
        self.debug = lambda s: self.logging.debug(f"\n{s}\n")
 
    # @abstractmethod
    def train(self, file_path):
        self.debug(f"Training tokenizer on file: {file_path}")
        if self.custom_tokens:
            self.init_custom_tokens(self.custom_tokens)
        tokens, info = self.encode(file_path)
        self.vocab = info
        self.debug(f"Tokenizer training completed")
        raise NotImplementedError

    @abstractmethod
    def encode(self, input) -> torch.tensor:
        """
        Inputs some data, returns a tensor (a 2d list of integers), i.e. the token encodings
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: torch.tensor):
        """
        Inputs a tensor (a 2d list of integers), returns a the expected datatype
        """
        raise NotImplementedError  

    def load_data(self, path: str) -> torch.Tensor:
        """
        Reads the input (either a string containing values or a string pointing towards a file) and returns a tensor
        """
        raise NotImplementedError

    def save_tokenizer(self, file_name: str) -> None:
        """
        Saves two files: file_name.model and file_name.vocab
        These can be used for loading and visualizing the model (e.g. the vocab + TODO) respectively
        .model can be used for the load() function
        """
        raise NotImplementedError
    
    def load_tokenizer(self, file_name: str):
        """
        Loads file_name.model, file_name has to be a direct reference to the correct .model file
        May work with .model files not saved by my code? TODO TEST
        """
        raise NotImplementedError
    
    def pad_input(self, input: torch.Tensor, length: int, dims: int) -> torch.Tensor:
        """
        Pads the input to the desired padding length
        """
        ### ChatGPT ###
        while input.size(dims-1) < length:
            input = F.pad(input, (0, 1), "constant", 0)
        self.debug(f"Tensor: {input}")
        returnal = input.masked_fill(torch.tril(torch.ones(input.size(dims-1), input.size(dims-1))) == 0, 0)
        self.debug(f"Returnal: {returnal}")

        return returnal
    
    def _get_byte_pairs(self, quantized_data):

        pairs = defaultdict(int)
        for row in quantized_data:
            for i in range(len(row) - 1):
                pair = (int(row[i]), int(row[i + 1]))
                pairs[pair] += 1
        return pairs

    def _merge_most_common(self, data, pair_to_merge, new_token):
        new_data = []
        for row in data:
            i = 0
            new_row = []
            while i < len(row):
                if i < len(row) - 1 and (row[i], row[i + 1]) == pair_to_merge:
                    new_row.append(new_token)
                    i += 2
                else:
                    new_row.append(row[i])
                    i += 1
            new_data.append(new_row)
        return new_data

    def byte_pair_encoding(self, quantized_data, vocab, merges): #n_merges is the amount of individual samples we're using; INCREASE THIS
        """
        Takes quantized data (TODO ADD EXAMPLE) and updates self.vocab
        """
        data = [list(row) for row in quantized_data]
        token_n = self.n_merges
        
        for i in range(self.n_merges): # Voor elke mogelijke waarde in vocab
            pairs = self._get_byte_pairs(np.array(data, dtype='object'))
            if not pairs:
                break
                
            most_common = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[most_common] = token_n # We gebruiken sample rate omdat dat exact de volgende stap is. 
            # in BPE, i.e. ((token_a, token_b): token_n)
            self.vocab[token_n] = most_common
            
            data = self._merge_most_common(data, most_common, token_n)
            token_n += 1

        self.debug(f"BPE: {self.vocab}")
        
        return data, {'vocab' : self.vocab, 'merges': self.merges}

    # Dict[str, Union[List[int], np.ndarray]]
    def init_custom_tokens(self, custom_tokens):
        token_to_pattern = {}
        pattern_to_token = {}
        vocab = {}
        next_token = 0
        # Initialize with base vocabulary
        
        # Add custom tokens 
        for token_name, pattern in custom_tokens.items():
            pattern_tuple = tuple(pattern)
            token_to_pattern[next_token] = pattern_tuple
            pattern_to_token[pattern_tuple] = next_token
            vocab[next_token] = pattern_tuple
            next_token += 1
        print(vocab)
        # Vocab is the initial bpe dict
        vocab = {i: i+next_token for i in range(sample_rate)}
        print(f"i:i+{vocab}")
        return vocab

    


@dataclass
class AudioTokenizer(GenericTokenizer):
    sample_rate:    int     = 512 # The amount of individual frames that the soundbyte is converted 
    sample_rate = 1500
    n_mels = 128
    n_fft = 1024
    hop_length = 512 
    n_merges=64 # Bepaalt dit hoeveel individuele tokens er zijn?
    n_bits=8 # Used in quantization

    # #TODO Hier moet je nog over nadenken
    # visualize: bool     = True
    # vocab: int          = 512    # SOMETHING -> Refers to the to the total amount of tokens, i.e. this is the initialization for the length of list of tokens (with their own context?)
    #                                 # Moet er een verbinding zijn tussen deze parameter en de input? -> I.e. moet ik dit uit de input halen of kan k gwn 3 invullen 
    #                                 # Verwachting H1: Relevanter nummer is beter, maar kan door bruteforcen achterkomen'
    
    # patch_embedding:int = 60  # Het aantal 'frames' in een sound segment; denk aan de sampling rate; in relatie met lengte segmenten
    #                                 # I.e, nu doen we dus 60 frames in elk segment van:
    # sound_length: int   = 1      # 1 seconden
    # n_mfcc: int         = 14        # number of MFCC features to extract (No I don't know what this means)
    # n_mels: int         = 128       # Number of mel bands for the mel-spectogram
    # n_fft: int          = 1024      # Number size of the FFT, determines the frequency resolution
    # hop_length: int     = 512       # Number of samples between frames
    # sample_rate: int    = 16000
    # max_length: int     = 1024     # Maximum sequence length for input tokens




    def __post_init__(self):
        return super().__post_init__()
    
    def encode(self, input):
        """
        Accepteert een bestandspad, returnt een tensor met tokens en de tokeninfo
        verwacht dat de tokenizer getraind is
        maakt gebruik van self.vocab om de tensor te maken
        gebruikt load_data() om data in te laden
        gebruikt extract_features() om de mel spec te laden 
        gebruikt quantize_features om de features te quantificeren
        gebruikt BPE om dit te tokenizeren
        """
        loaded_data, or_sample_rate         = self.load_data(input)
        features_of_data    = self.extract_features(loaded_data, or_sample_rate)
        quantized_data      = self.quantize_features(features_of_data)
        encoded_data, info  = self.byte_pair_encoding(quantized_data)
        if self.debug:
            self.debug(f"Byte pair encoded information: {info}")
        return encoded_data, info

        # return super().encode(input)

    def decode(self, tensor, path) -> str:
        """
        Takes a tensor and decodes it, returns the saved folder string 
        """
        return NotImplementedError

    def extract_features(self, waveform, or_sample_rate):
        
        num_tokens = waveform.size(1)  # Assuming waveform shape is (channels, time-steps)
        # Convert stereo to mono
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if or_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=or_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / torch.max(torch.abs(waveform)) # Ik vraag me af of dit waarde heeft
        MS_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_spec = MS_transform(waveform)
        mel_spec = mel_spec.squeeze().transpose(0, 1).numpy()
        # Reshapes to a 2d array, I don't know what it was though
        mel_spec = mel_spec.reshape(-1, self.n_mels)
        return mel_spec

    def quantize_features(self, mel_spec):
        min_val = np.min(mel_spec)
        max_val = np.max(mel_spec)
        
        quantized = np.round((mel_spec - min_val) / (max_val - min_val) * (2**self.n_bits - 1))
        return quantized.astype(np.uint8)

    def plot(self, waveform):
        # Convert to numpy for visualization
        features_np = waveform.squeeze().numpy()
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot mel spectrogram
        img = librosa.display.specshow(
            features_np,
            y_axis='mel',
            x_axis='time',
            ax=ax1
        )
        ax1.set_title('Mel Spectrogram (normalized)')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')
        
        # Plot a few mel frequency bands over time
        times = np.arange(features_np.shape[1])
        for i, band in enumerate([0, 20, 40, 60]):
            if band < features_np.shape[0]:
                ax2.plot(times, features_np[band], label=f'Mel Band {band}')
        ax2.set_title('Selected Mel Bands Over Time')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Normalized Amplitude')
        ax2.legend()
        
        plt.tight_layout()
        return plt

    def load_data(self, path: str) -> torch.Tensor:
        """
        Reads the input (either a string containing text or a string pointing towards a file/folder) and returns a tensor
        If an empty dir is presented, returns a 3x3 tensor of 0s
        """
        self.debug(path)
        if os.path.isfile(path):

            waveform, or_sample_rate = torchaudio.load(path) 
            return waveform, or_sample_rate
        if os.path.isdir(path):
            to_concat = []
            for file in os.listdir(path):
                waveform, or_sample_rate = torchaudio.load(os.path.join(path, file))
                to_concat.append(waveform)
            if to_concat == []:
                return torch.zeros((3, 3), dtype=torch.long), 5
            return torch.cat(to_concat), or_sample_rate
        raise AttributeError 





class TextTokenizer (GenericTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distinct_tokens = sorted(list(set(self.data)))
        self.num_distinct_tokens = len(self.distinct_tokens)
        self.stringtoint = { ch:i for i,ch in enumerate(self.distinct_tokens) } # {"A":0, "B":1, ..., "!": 80}
        self.inttostring= { i:ch for i,ch in enumerate(self.distinct_tokens) } # {0:"A", 1:"B", ..., 80:"!"}

    def encode(self, input: str) -> list:



        
        return torch.tensor([self.stringtoint[c] for c in input], dtype=torch.long) # MVP encoding algo
    
    def decode(self, input: list) -> str:
        return ''.join([self.inttostring[i] for i in input]) # MVP encoding algo
    
    def load_data(self, path: str) -> torch.Tensor:
        """
        Reads the input (either a string containing text or a string pointing towards a file/folder) and returns a tensor
        """
        if os.path.isfile(path):
            return self._load_data(path)
        if os.path.isdir(path):
            return torch.cat([self._load_data(file) for file in os.listdir(path)]) 
        raise AttributeError

    def pad_input(self, input: torch.Tensor, mask_length: int=16, dims: int=1) -> torch.Tensor:
        """
        Default naar 16 mask length en 2 dims ()
        """
        return super().pad_input(input, mask_length, dims)