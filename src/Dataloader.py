import os
from dataclasses import dataclass

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class Dataloader():
    file_path: str
    isdir = False
    loaded = []
    transformed = []
    """
    Kan een bestand meekrijgen ook bij initialisatie zowel als .load() voor dubbele bruikbaarheid,
    Maakt de code van .load() iets wonkyer
    """
    def __post_init__(self):
        pass

    def load(self, file_path=None):
        filepath = os.Path(self.file_path if file_path == None else file_path)
        if not filepath.exists():
            raise FileNotFoundError(f"file not found: {filepath}")
        # TODO: verfwijder alles en vervang het met:
        # als bestand:
            # return iterable met 1 item(Laad bestand)
        # andersals folder:
        #     voor bestand in folder:
        #         laad bestand
        #     return alle als individuele items in een iterable
    
        if os.path.isdir(filepath): 
            for secondfile in os.listdir(filepath):
                if secondfile.isdir():
                    # for thirdfile in 
            self.loaded = [self.load_file(file) for file in filepath.listdir()]
        else:
            self.loaded.append(self.load_single_file(filepath))
        
    def load_single_file(self, filepath):
        fl = FileLoader(filepath)
        loaded = fl.load()
        return loaded



        



class FileLoader:
    """A utility class for loading different types of files."""
    
    def __init__(self, filepath):
        self.filepath = os.Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
    def load(self):
        suffix = self.filepath.suffix.lower()
        
        if suffix == '.txt':
            return self.load_text()
        elif suffix == '.json':
            return self.load_json()
        elif suffix == '.csv':
            return self.load_csv()
        elif suffix in ['.yaml', '.yml']:
            return self.load_yaml()
        elif suffix == '.wav' and TORCH_AVAILABLE:
            return self.load_torchaudio()
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
    
    def load_text(self):
        """Load a text file."""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_json(self):
        """Load a JSON file."""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def load_csv(self):
        """Load a CSV file."""
        result = []
        with open(self.filepath, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                result.append(row)
        return result
    
    def load_yaml(self):
        """Load a YAML file."""
        with open(self.filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def load_torchaudio(self):
        """Load an Excel file using pandas."""
        if not TORCH_AVAILABLE:
            raise ImportError("pandas is required to load Excel files")
        return torchaudio.load(self.filepath)
    

