import torch
from tokenizers import Tokenizer
from pathlib import Path

from models.model import build_transformer
from models.config import get_config 
from models.config import get_weights_file_path 


# Global variables to hold the loaded model and tokenizers
MODEL = None
TOKENIZER_SRC = None
TOKENIZER_TGT = None
CONFIG = None

def load_model():
    """Loads the model and tokenizers, running only once."""
    global MODEL, TOKENIZER_SRC, TOKENIZER_TGT, CONFIG

    if MODEL is not None:
        print("Model and tokenizers already loaded.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    CONFIG = get_config()
    CONFIG['preload'] = 'latest' # Or whatever you use to specify the model file
    CONFIG['seq_len'] = 350 # Ensure this matches your trained model
    
    # Define asset paths
    tokenizer_path_src = Path(__file__).parent.parent / "assets" / CONFIG['tokenizer_file'].format(CONFIG['lang_src'])
    tokenizer_path_tgt = Path(__file__).parent.parent / "assets" / CONFIG['tokenizer_file'].format(CONFIG['lang_tgt'])

    # Load tokenizers
    TOKENIZER_SRC = Tokenizer.from_file(str(tokenizer_path_src))
    TOKENIZER_TGT = Tokenizer.from_file(str(tokenizer_path_tgt))

    # Build model
    MODEL = build_transformer(
        TOKENIZER_SRC.get_vocab_size(),
        TOKENIZER_TGT.get_vocab_size(),
        CONFIG["seq_len"],
        CONFIG["seq_len"],
        d_model=CONFIG["d_model"]
    ).to(device)

    model_filename_relative = get_weights_file_path(CONFIG, "19")

    # Adjust the path to be relative to the project root, not the current file
    model_filename = Path(__file__).parent.parent / "assets"/ model_filename_relative

    if model_filename.exists():
        print(f"Preloading model from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        MODEL.load_state_dict(state['model_state_dict'])
    else:
        print(f"WARNING: Model weights not found at {model_filename}. Using a randomly initialized model.")

    MODEL.eval() # Set model to evaluation mode

# Call this function once when the application starts
load_model()