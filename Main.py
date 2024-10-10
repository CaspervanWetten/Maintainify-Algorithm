# This is basically the "quick" file
from Interface import Transformer

# Verwachtte interface:


data_folder = "Data/1-18527-A-44.wav"
Data = "Data/"
transformer = "folder_path/tokenizer.vocab"
dog_sound = "Data/dog2.wav"
T = Transformer(tokenizer=dog_sound, debug_bool=True)
# T.load_model("Models/test.pt")
# T.load_data(data_folder)

# print(T)

cat_data = "Data/categorized/"
T.load_categorized_data(cat_data)

T.optimize_categorization()
print(f"categorized: {T.generate(dog_sound)}")