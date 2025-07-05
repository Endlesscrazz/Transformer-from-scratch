import torch
import torch.nn.functional as F
from models.dataset import causal_mask
from app.model_handler import MODEL, TOKENIZER_SRC, TOKENIZER_TGT, CONFIG

def translate(sentence: str):
    """
    Translates a source sentence into the target language and extracts attention scores.
    """
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MODEL.to(device)
        model.eval()

        # Pre-process the source sentence
        src_tokens = TOKENIZER_SRC.encode(sentence).ids
        sos_token = TOKENIZER_SRC.token_to_id('[SOS]')
        eos_token = TOKENIZER_SRC.token_to_id('[EOS]')
        pad_token = TOKENIZER_SRC.token_to_id('[PAD]')

        enc_input_tokens = [sos_token] + src_tokens + [eos_token]
        enc_num_padding_tokens = CONFIG['seq_len'] - len(enc_input_tokens)
        if enc_num_padding_tokens < 0:
            raise ValueError("Source sentence is too long.")

        encoder_input = torch.tensor(enc_input_tokens + [pad_token] * enc_num_padding_tokens, dtype=torch.int64).unsqueeze(0).to(device)
        encoder_mask = (encoder_input != pad_token).unsqueeze(1).unsqueeze(1).int().to(device)

        # Pre-compute the encoder output
        encoder_output = model.encode(encoder_input, encoder_mask)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.tensor([[TOKENIZER_TGT.token_to_id('[SOS]')]], dtype=torch.int64).to(device)

        # Store all cross-attention scores for each generated token
        all_cross_attention_weights = []

        # Decoding loop
        while decoder_input.size(1) < CONFIG['seq_len']:
            # Create mask for the current decoder input
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

            # Calculate output
            # The modified decoder now returns attention scores
            out, cross_attention_scores = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

            # Project to vocabulary and get the last token
            prob = model.project(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            
            # Append the new token to the decoder input
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_word_idx.item()]], dtype=torch.int64).to(device)], dim=1)

            # Store the cross-attention weights for the token we just generated
            # We take the attention from the last layer for simplicity: cross_attention_scores[-1]
            # Shape: (batch=1, heads, new_token_pos=1, src_len) -> (heads, src_len)
            last_layer_attention = cross_attention_scores[-1].squeeze(0)[:, -1, :].cpu().numpy().tolist()
            all_cross_attention_weights.append(last_layer_attention)

            if next_word_idx.item() == TOKENIZER_TGT.token_to_id('[EOS]'):
                break

        # Post-process the output
        decoded_tokens = decoder_input.squeeze(0).cpu().numpy()
        translated_text = TOKENIZER_TGT.decode(decoded_tokens)
        
        # Get tokens for visualization labels
        source_tokens_for_viz = [TOKENIZER_SRC.id_to_token(id) for id in enc_input_tokens]
        target_tokens_for_viz = [TOKENIZER_TGT.id_to_token(id) for id in decoded_tokens]
        
        # Structure the response
        response = {
            "translated_text": translated_text,
            "source_tokens": source_tokens_for_viz,
            "target_tokens": target_tokens_for_viz,
            # Attention shape: (target_len, num_heads, source_len)
            "attention_weights": all_cross_attention_weights
        }
        return response