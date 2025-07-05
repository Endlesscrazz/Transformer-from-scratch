import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    """Yields all sentences from the dataset for a specific language."""
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(ds, lang, output_path):
    """Builds and trains a new tokenizer."""
    print(f"Building tokenizer for language: '{lang}'...")
    # Initialize a new tokenizer with WordLevel model
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    # Use whitespace to split words
    tokenizer.pre_tokenizer = Whitespace()
    # Define the trainer with special tokens
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    
    # Train the tokenizer
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    
    # Save the tokenizer to the specified path
    tokenizer.save(str(output_path))
    print(f"Tokenizer for '{lang}' saved to {output_path}")
    return tokenizer

def main():
    # --- Configuration ---
    lang_src = 'en'
    lang_tgt = 'hi'
    
    # --- Create destination folder ---
    output_dir = Path("assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_path_src = output_dir / f"tokenizer_{lang_src}.json"
    tokenizer_path_tgt = output_dir / f"tokenizer_{lang_tgt}.json"

    # --- Load a small part of the dataset ---
    # Using 30,000 examples for POC
    print("Loading a subset of the wmt14 hi-en dataset...")
    ds_raw = load_dataset('wmt14', "hi-en", split='train', streaming=False)

    ds_subset = ds_raw.select(range(30000)) 
    print(f"Dataset subset loaded with {len(ds_subset)} examples.")

    # --- Build and Save Tokenizers ---
    if not tokenizer_path_src.exists():
        build_tokenizer(ds_subset, lang_src, tokenizer_path_src)
    else:
        print(f"Tokenizer already exists at {tokenizer_path_src}, skipping.")

    if not tokenizer_path_tgt.exists():
        build_tokenizer(ds_subset, lang_tgt, tokenizer_path_tgt)
    else:
        print(f"Tokenizer already exists at {tokenizer_path_tgt}, skipping.")
        
    print("\nTokenizer generation complete.")
    print(f"Your files are located in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()