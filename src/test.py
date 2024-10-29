


from Tokenizer import TokenizerInterface
from Config import Config

config = Config()
T = TokenizerInterface(config, passed_tokenizer="KMeansTokenizer")

a = T.encode()

print(T)