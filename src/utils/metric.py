import re
from jiwer import wer
import torch

def clean_text(text):
    """
    Clean prediction text by removing 'Human' at the end or '**'.
    """
    # Remove 'Human' at the end (with or without punctuation)
    text = re.sub(r'(?:\s*Human[\s\.\!\?]*)$', '', text)
    # Remove ** anywhere in the sentence
    text = re.sub(r'\*\*', '', text)
    # Remove redundant whitespace
    return text.strip()

def compute_wer(tokenizer, pad_outputs, pad_targets, ignore_label):
    """
    Calculate Word Error Rate (WER) using masking instead of slicing.

    Args:
        tokenizer: Tokenizer with .batch_decode() method.
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id, usually -100.

    Returns:
        float: Average WER for all sequences (0.0 - 1.0).
    """
    mask = pad_targets != ignore_label

    masked_outputs = []
    masked_targets = []

    for i in range(pad_outputs.size(0)):
        valid_output = pad_outputs[i][mask[i]].tolist()
        valid_target = pad_targets[i][mask[i]].tolist()

        masked_outputs.append(valid_output)
        masked_targets.append(valid_target)

    pred_texts = tokenizer.batch_decode(masked_outputs, skip_special_tokens=True)
    print("PREDCIT", pred_texts)
    
    target_texts = tokenizer.batch_decode(masked_targets, skip_special_tokens=True)
    print("TARGET", target_texts)

    total_wer = 0.0
    for pred, ref in zip(pred_texts, target_texts):
        cleaned_pred = clean_text(pred)
        cleaned_ref = clean_text(ref)
        total_wer += wer(cleaned_ref, cleaned_pred)

    return total_wer / len(pred_texts) if pred_texts else 0.0
