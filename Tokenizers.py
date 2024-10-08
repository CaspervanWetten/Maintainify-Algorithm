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
from dataclasses import dataclass
from collections import defaultdict




@dataclass
class GenericTokenizer(ABC):
    # Other attribute
    debug:        bool    = False

    def __post_init__(self) -> None:
        self.logging = logging.getLogger(__name__); logging.basicConfig(level=logging.DEBUG if self.debug else logging.WARNING)
        pass



        # [setattr(self, key, value) for key, value in kwargs.items()] # Arbitrarily accept all keywords passed
        # if kwargs.get("data", None):
        #     self.data = kwargs['data']
        # else:
        #     raise AttributeError("No data passed!")
        # logging.basicConfig(level=logging.DEBUG if kwargs.get("debug", None) != None else logging.WARNING)
        self.loging = logging.getLogger(__name__)
        self.debug = lambda s: self.loging.debug(f"\n{s}\n")

 
    # @abstractmethod
    def train(self, **kwargs):
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

    @abstractmethod # Do just super() for both but have different defaults, have this have no defaults
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
    

@dataclass
class AudioTokenizer(GenericTokenizer):
    #TODO Hier moet je nog over nadenken
    visualize: bool     = True
    vocab: int          = 512    # SOMETHING -> Refers to the to the total amount of tokens, i.e. this is the initialization for the length of list of tokens (with their own context?)
                                    # Moet er een verbinding zijn tussen deze parameter en de input? -> I.e. moet ik dit uit de input halen of kan k gwn 3 invullen 
                                    # Verwachting H1: Relevanter nummer is beter, maar kan door bruteforcen achterkomen'
    
    patch_embedding:int = 60  # Het aantal 'frames' in een sound segment; denk aan de sampling rate; in relatie met lengte segmenten
                                    # I.e, nu doen we dus 60 frames in elk segment van:
    sound_length: int   = 1      # 1 seconden
    n_mfcc: int         = 14        # number of MFCC features to extract (No I don't know what this means)
    n_mels: int         = 128       # Number of mel bands for the mel-spectogram
    n_fft: int          = 1024      # Number size of the FFT, determines the frequency resolution
    hop_length: int     = 512       # Number of samples between frames
    sample_rate: int    = 16000
    max_length: int     = 1024     # Maximum sequence length for input tokens



    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, file_path) -> torch.tensor:
        """
        Accepts an audio signal and returns a properly tokenized version
        Audio is pre-processed (normalized, etc)
        Audio is feature extracted (MFCC, CQT, melspectograms)
        Audio is Quantized (whatever that means)
        Batched?
        """
        # Na het laden van de audio moet het dus sws eerst in een lijst per frames met x seconden gehakt worden
        # Dit wordt dan opgehakt in kleine frames (*letter*lijk dit zijn tokens)
        # uit de frames lezen we belangrijke data
        # Dit is een tensor
        # Daar wordt mee gerekend.

        self.waveform, sample_rate = torchaudio.load(file_path) # Test 1: This is a very smallll tensor
        
        # Convert stereo to mono
        if self.waveform.shape[0] > 1: 
            self.waveform = torch.mean(self.waveform, dim=0, keepdim=True)
        # Resample:
        if sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            self.waveform = resampler(self.waveform)
        
        # Actual feature extraction:
            # Normalize the waveform
        self.waveform = self.waveform / torch.max(torch.abs(self.waveform))

            # Mel spectogram
        MS_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, normalized=True)
        mel_spec = MS_transform(self.waveform)
        self.debug(f"mel_spec after creation: {mel_spec}")
            # Convert to decibel
        db_mel_spec = T.AmplitudeToDB()(mel_spec)
            # Normalize DB mel_spec
        db_mel_spec_normalized = (db_mel_spec - db_mel_spec.mean()) / db_mel_spec.std()

        self.debug(db_mel_spec_normalized)
  
    #    # MFCC
    #     MFCC_transform = T.MFCC(sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, melkwargs={
    #         "n_fft": self.n_fft,
    #         "hop_length": self.hop_length,
    #         "n_mels": self.n_mels,
    #         "center": True
    #     })
    #     mfcc = MFCC_transform(self.waveform)
     
        # Print out the waveform shape and sample rate for confirmation
        self.debug(f"min: {db_mel_spec_normalized.min()}")
        self.debug(f" max: {db_mel_spec_normalized.max()}")
        self.debug(f"mean: {db_mel_spec_normalized.mean()}")
        self.debug(f"mel_spec after processing: {db_mel_spec_normalized}")
        return db_mel_spec_normalized
 
    def load_data(self, path: str) -> torch.Tensor:
        """
        Reads the input (either a string containing text or a string pointing towards a file/folder) and returns a tensor
        If an empty dir is presented, returns a 3x3 tensor of 0s
        """
        if os.path.isfile(path):
            print("x")
            return self.encode(path)
        if os.path.isdir(path):
            to_concat = [self.encode(os.path.join(path, file)) for file in os.listdir(path)]
            if to_concat == []:
                return torch.zeros((3, 3), dtype=torch.long)
            return torch.cat(to_concat)
        raise AttributeError 
    
    def decode(self, tensor, path) -> str:
        """
        Takes a tensor and decodes it, returns the saved folder string 
        """
        return NotImplementedError

    def preprocess(self, source: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582,
        ) -> torch.Tensor:
            fbanks = []
            for waveform in source:
                waveform = waveform.unsqueeze(0) * 2 ** 15
                fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
                fbanks.append(fbank)
            fbank = torch.stack(fbanks, dim=0)
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            return fbank 
    
    def quantize_features(self, mel_spec, n_bits=8):
        min_val = np.min(mel_spec)
        max_val = np.max(mel_spec)
        
        quantized = np.round((mel_spec - min_val) / (max_val - min_val) * (2**n_bits - 1))
        return quantized.astype(np.uint8)

    def get_byte_pairs(self, quantized_data):

        pairs = defaultdict(int)
        for row in quantized_data:
            for i in range(len(row) - 1):
                pair = (int(row[i]), int(row[i + 1]))
                pairs[pair] += 1
        return pairs

    def merge_most_common(self, data, pair_to_merge, new_token):
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

    # def byte_pair_encoding(quantized_data, n_merges): #n_merges is the amount of individual samples we're
    #     vocab = {i: i for i in range(256)}
    #     next_token = 256
    #     data = [list(row) for row in quantized_data]
    #     merges = {}
        
    #     for i in range(n_merges):
    def byte_pair_encoding(self, quantized_data, sample_rate: int): #n_merges is the amount of individual samples we're using; INCREASE THIS
        vocab = custom_tokens(sample_rate) # Initialize a vocab dict the length of sample rate
        data = [list(row) for row in quantized_data]
        token_n = sample_rate
        merges = {}
        
        for i in range(sample_rate): # Voor elke mogelijke waarde in vocab
            pairs = get_byte_pairs(np.array(data, dtype='object'))
            if not pairs:
                break
                
            most_common = max(pairs.items(), key=lambda x: x[1])[0]
            merges[most_common] = token_n # We gebruiken sample rate omdat dat exact de volgende stap is. 
            # in BPE, i.e. ((token_a, token_b): token_n)
            vocab[token_n] = most_common
            
            data = merge_most_common(data, most_common, token_n)
            token_n += 1
        
        return data, {'vocab': vocab, 'merges': merges}

    def extract_audio_features(self, file_path):
        sample_rate = 1500
        n_mels = 128
        n_fft = 1024
        hop_length = 512 
        n_merges=64 # Bepaalt dit hoeveel individuele tokens er zijn?

        waveform, or_sample_rate = torchaudio.load(file_path,) 
        num_tokens = waveform.size(1)  # Assuming waveform shape is (channels, time-steps)
        # Convert stereo to mono
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if or_sample_rate != sample_rate:
            resampler = T.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / torch.max(torch.abs(waveform)) # Ik vraag me af of dit waarde heeft
        MS_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spec = MS_transform(waveform)
        mel_spec = mel_spec.squeeze().transpose(0, 1).numpy()
        print(f"transposed and numpyfied: {mel_spec}")
        # Reshapes to a 2d array, I don't know what it was though
        mel_spec = mel_spec.reshape(-1, n_mels)
        print(f"reshaped: {mel_spec}")

        quantized_data = quantize_features(mel_spec)
        print(f"quantized: {quantized_data}")
        encoded_data, bpe_info = byte_pair_encoding(quantized_data, n_merges)
        

        # mel_spec = torch.flatten(mel_spec, start_dim=0, end_dim=1)
        # long_spec = float_to_long_tensor(mel_spec)

        return encoded_data, num_tokens, bpe_info


    # Dict[str, Union[List[int], np.ndarray]]
    def custom_tokens(self, sample_rate):
        custom_tokens = {
            "dog": "dog",
            "cat": "cat",
            "elephant": "elephant",
        } # DIt wordt een json 
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


    def forward_padding_mask(self, generic_tensor: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return super().forward_padding_mask(generic_tensor, padding_mask)

    def pad_input(self, input: torch.Tensor, length: 32, dims: 3) -> torch.Tensor:
        return super().pad_input(input, length, dims)
    
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