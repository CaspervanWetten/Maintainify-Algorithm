

from Config import Config
from Logger import Logger
from Model import TransformerModel
from Tokenizer import TokenizerInterface
from Dataloader import DataloaderInterface
from dataclasses import dataclass


@dataclass
class Transformer():
    # Attributes:
    example_data: str   # Moet een pad naar een bestand zijn

    tokenizer: str  = ""    # Welk tokenizatie algo wil je gebruiken?
    dataloader: str = ""    # Welk data-laad algo wil je gebruiken
    debug: bool     = False
    train_data      = None #TODO: Hoe is dit anders als het categorizatie data is?
    test_data       = None
    val_data        = None
    config: Config  = Config()
    logger: Logger  = Logger()
    model: TransformerModel = TransformerModel()

    def __post_init__(self) -> None:
        