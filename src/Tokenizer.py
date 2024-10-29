
import torch

import numpy as np

from abc import ABC
from Config import Config
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenericTokenizer(ABC):
    config: Config

    def encode(self):
        return "encoded"
    
    def decode(self):
        return "decoded"

@dataclass
class KMeansTokenizer(GenericTokenizer):
    def encode(self, to_encode=None):
        """
        TODO: Omschrijf hoe de verwachtte to_encode eruitziet
        Ik gok een np.array of torch.tensor
        """
        if to_encode == None:
            return torch.zeros((16, 16), dtype = torch.long)
        cluster_values, tokens = self.k_means_clustering(to_encode)
        return tokens
    
    def k_means_clustering(self, to_cluster):
        input_vocab = self.config.input_vocab
        max_iters = self.config.k_means.max_iters
        convergence_tolerance = self.config.k_means.convergence_tolerance
        centroids = np.random.choice(to_cluster, size=input_vocab, replace=False)
        for _ in range(max_iters):
            distances = np.abs(to_cluster[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([to_cluster[labels == k].mean() for k in range(input_vocab)])
            if np.all(np.abs(new_centroids - centroids) < convergence_tolerance):
                break
            centroids = new_centroids
        return centroids, labels

@dataclass
class BPETokenizer(GenericTokenizer):
    pass

@dataclass
class TokenizerInterface():
    config: Config 
    example_data: str = ""
    passed_tokenizer: str = Optional[None]
    _tokenizer: GenericTokenizer = Optional[None]

    def __post_init__(self):
        _available_tokenizers = {"KMeansTokenizer" : KMeansTokenizer, "BPETokenizer": BPETokenizer}
        
        if self.passed_tokenizer is not None:
            if self.passed_tokenizer in _available_tokenizers.keys():
                self._tokenizer = _available_tokenizers[self.passed_tokenizer](self.config)
            else:
                raise ValueError(f"Passed tokenizer not an implemented algorithm\nChoose between {_available_tokenizers}")
        elif self.example_data != "":
            self._tokenizer = self._select_tokenizer(self.example_data)
        else:
            raise ValueError("Neither a tokenizer nor example data were provided")
        
    def encode(self, to_encode=None) -> torch.Tensor:
        """
        Encodes the passed the argument using the selected tokenizer, returns a Tensor
        """
        return self._tokenizer.encode(to_encode)
    
    def decode(self, to_decode):
        """
        Decodes the passed the tensor using the selected tokenizer, returns an instance
        """
        return self._tokenizer.decode(to_decode)
