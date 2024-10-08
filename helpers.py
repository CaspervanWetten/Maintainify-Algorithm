# File to keep the other files cleaner, customs to suit my needs



"""
gets input.txt if not from this datasets folder, from the datasets folder in ../Code
"""
def get_input(inp = "shakespeare"):
    if inp.lower() == "shakespeare":
        try:
            with open('Data/input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            with open('Code/Data/input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        return text

    if inp.lower() == "wouter":
        try:
            with open('Data/wouter.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            with open('Code/Data/wouter.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        return text
    
    raise FileNotFoundError("Bestand niet gevonden!")