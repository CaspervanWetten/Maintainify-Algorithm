# This is basically the "quick" file
from Interface import Transformer

# Verwachtte interface:


data_folder = "Data"

T = Transformer("audio")
# T.load_model("Models/test.pt")
# T.load_data(data_folder)

print(T)
print(T.generate())