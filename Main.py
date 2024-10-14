# This is basically the "quick" file
from Interface import Transformer
import torchaudio

# Verwachtte interface:


data_folder = "Data/1-18527-A-44.wav"
Data = "Data/"
transformer = "folder_path/tokenizer.vocab"
dog_sound = "Data/dog2.wav"
T = Transformer(tokenizer=dog_sound, debug_bool=True)
print(f"transformer stats: {T}")

categorization_folder = "Data\\categorized\\"
T.load_categorized_data(categorization_folder)
T.optimize_categorization()



print(f"categorized: {T.generate(dog_sound)}")