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


# From Claude
import torch
def float_to_long_tensor(float_tensor, method='minmax', num_bins=1000):
    """
    Convert float tensor to long tensor using different methods
    
    Parameters:
    - float_tensor: input tensor with float values
    - method: 'minmax', 'rank', or 'quantile'
    - num_bins: number of discrete values in the output
    
    Returns:
    - tensor with dtype torch.long
    """
    if method == 'minmax':
        # Method 1: Min-Max scaling
        min_val = float_tensor.min()
        max_val = float_tensor.max()
        
        # Scale to 0...num_bins-1
        scaled = ((float_tensor - min_val) / (max_val - min_val) * (num_bins - 1)).long()
        
        return scaled
    
    elif method == 'rank':
        # Method 2: Rank-based conversion
        flat = float_tensor.flatten()
        sorted_indices = flat.argsort()
        ranks = torch.empty_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(flat), device=float_tensor.device)
        
        # Scale ranks to 0...num_bins-1
        scaled_ranks = (ranks.reshape(float_tensor.shape) * (num_bins - 1) // (float_tensor.numel() - 1)).long()
        
        return scaled_ranks
    
    elif method == 'quantile':
        # Method 3: Quantile-based binning
        flat = float_tensor.flatten()
        quantiles = torch.quantile(flat, torch.linspace(0, 1, num_bins, device=float_tensor.device))
        
        # Assign each value to a bin
        bins = torch.bucketize(float_tensor, quantiles).long() - 1
        bins = torch.clamp(bins, 0, num_bins - 1)
        
        return bins
    
    else:
        raise ValueError(f"Unknown method: {method}")