# This is basically the "quick" file
from Interface import Transformer

# Verwachtte interface:


data_folder = "Data"
transformer = "folder_path/tokenizer.vocab"
T = Transformer(tokenizer="Data\\1-22882-A-44.wav", debug_bool=True)
# T.load_model("Models/test.pt")
# T.load_data(data_folder)

# print(T)
print(f"generated: {T.generate(data_folder)}")