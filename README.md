# Transformer

## About The Project

1) Implementation of Transformer architecture,
2) Training on Anki Eng-Rus MT dataset for translation task.

## Getting Started

File to run:

    /executor/executor.py 

MlFlow logs are in: 

    executor/mlruns.rar

Visualization of accuracy on the training and test samples, loss are in: 

    saved_files/plots/
    
Visualization of Levenshtein distance on inference: 

    saved_files/plots/inference_levenshtein.PNG

on the y-axis - the distance averaged over all previous steps, on the x-axis - the inferential number of sentences. (For ~1000 offers average = ~96)

## Implementation details

The following modules have been implemented:


● OOP realization via nn.Model:
○ class Transformer
■ class Encoder
● class “encoder layer”
■ class Decoder
● class “decoder layer”
○ class “self-attention”
■ class “multi head attention”
■ class “masked scaled dot-product
attention”
○ class “Skip-connection”
○ class “LayerNorm”
■ class “Positional Encoding”
■ class “Embedding”
○ class AnkiDataset
■ Class BPE
○ class AnkiBPEDataloader
● PreNorm
● Training pipeline
● Greedy inference pipeline
● Beam search inference pipeline
● Minor improvements (label smoothing, initialization+normalization)
● BLEU score calculation
● Batch masks for padded sentences and decoder target input
