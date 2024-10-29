import os


try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Dataloader():
    """
    Loads the file path passed to it.
    Loads the file if file path is a file
    Loads the folder, file by file, if file path is a folder
    If the second level files are all folders, loads the data categorized by the folder names
    """
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.isdir = False
        self.loaded = []
        self.transformed = []

    def load(self, file_path=None):
        filepath = os.Path(self.file_path if file_path == None else file_path)
        if not filepath.exists():
            raise FileNotFoundError(f"file not found: {filepath}")
        if os.path.isdir(filepath): 
            for secondfile in os.listdir(filepath):
                if secondfile.isdir():
                    for thirdfile in 
            self.loaded = [self.load_file(file) for file in filepath.listdir()]
        else:
            self.loaded.append(self.load_single_file(filepath))
        
    def load_single_file(self, filepath):
        fl = FileLoader(filepath)
        loaded = fl.load()
        return loaded


    def transform(self, algorithm):
        possible_algos = {
            "mel_spec": self.mel_spec
        }
        if algorithm in possible_algos.keys():
            for loaded in self.loaded:
                self.transformed.append(possible_algos[algorithm](loaded))
        else:
            raise ValueError("Unknown algorithm")

    def mel_spec(self, loaded):
        waveform, or_sample_rate = loaded
        # Convert stereo to mono
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if or_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=or_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / torch.max(torch.abs(waveform)) # Ik vraag me af of dit waarde heeft
        MS_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_spec = MS_transform(waveform)
        # Turn it into a numpy array!
        mel_spec = mel_spec.squeeze().numpy().flatten()
        # convert to tensor
        return mel_spec

        



class FileLoader:L
    """A utility class for loading different types of files."""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
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
    

