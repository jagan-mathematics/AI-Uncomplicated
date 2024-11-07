# AI-Case-Studies

This repository is dedicated to a comprehensive exploration of the Transformer architecture, focusing on detailed insights into each component, its requirements, and various applications.

## Repo Setup
1) Set up the environment by running the following command:
   ```bash
   sh scripts/setup_env.sh
   ```

## Repository Structure
```
project
│   README.md
│   requirements.txt
│___scripts (contains setup and other script files)
│___core
|   |____tokenizer (holds all tokenizer training files and artifacts)
|   |____activation (holds all activation implementation and test cases)
|   |____configuratioin (hold all model configuration)
|   |____layers (hold all custom transformers and other model layers)
|   |____utils (hold all helper utils)
│___study (contains other experimentation studies and other resources)
|
```

## Transformers
This repository provides an in-depth implementation of the Transformer model's decoder architecture, with components focused on tokenization, training, and customization. The repository serves as a hands-on resource for experimenting with state-of-the-art models, making it suitable for those interested in advanced AI applications.


<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Mt09UTRNbV88dvl8mJ8NzQ.png" alt="transformer architecture" style="width:50%;"/>



## Studies
The following Jupyter notebooks explore various aspects of tokenization:

### Tokenizer
This repository includes resources for training and customizing Byte Pair Encoding (BPE) tokenizers, essential for handling text input efficiently in Transformer models.

- `AI-Uncomplicated/study/Tokenizer.ipynb`: Overview and insights on tokenization techniques.
- `AI-Uncomplicated/study/tokenizer_training_toy.ipynb`: Hands-on guide for training a tokenizer with toy datasets.

#### Training and Customization
For advanced users interested in training and modifying BPE tokenizers:

- **Training a BPE Tokenizer**: Use `AI-Uncomplicated/tokenizer/bpe/trainer.py` to train a tokenizer from scratch.
- **Post-Processing (Token Addition & Removal)**: The `AI-Uncomplicated/tokenizer/bpe/notebooks/post_process_trained_tokenizer.ipynb` notebook provides tools for adding or removing specific tokens from an existing tokenizer.

### Position Encoding

In Transformer models, position encoding is crucial for providing a sense of word order in sequences since these models lack inherent positional awareness. This section explores both absolute and relative position encoding techniques.

- **What is Positional Encoding?**  
  To grasp the fundamentals of positional encoding and sinusoidal encoding, explore the notebook:  
  `AI-Uncomplicated/study/sinisouidal_encoding.ipynb`.  
  This provides a detailed explanation of how positional information is encoded mathematically in Transformers.
- **Relative Positional Encoding**  
  Relative positional encoding introduces flexibility by enabling Transformers to consider relationships between tokens rather than absolute positions. Several new approaches have been proposed to implement this effectively.  
  Use the notebook `AI-Uncomplicated/study/rope_positional_encoding.ipynb` to delve into the *Rotary Positional Embedding (ROPE)* method, understand its motivation, and see how it works in practice.


---
**References**  
[Attention is All You Need](https://arxiv.org/abs/1706.03762)  
[Neural Machine Translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)