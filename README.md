# Jabarti Project (LLM Training Project From Scratch using PyTorch)

## Phase 1 - Dataset Exploration

[The used datatest](https://huggingface.co/datasets/bakrianoo/jabarti-llm-dataset)

[Another dataset that could be used to build your own experience or anyother preferable one](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)

### Pre-training

While exploring and prepraing the dataset for pre-training, we've to look at two things:

1. The language - The language must be clear, we mean by that the basics of the language must be correct and met with the dataset, like the grammar, the singular, the tenses, etc... for any language that you want to you want to train the model on (in our case it would be multilingual in both Arabic and English).

2. The knowledge - What's the range of knowledge that you want to cover by training your model in the pre-training phase? All the use topics (Cooking, Scicology, Medicine, History, Football, Coding, Management, IT, etc.) - this is the most common? Or some use topics only?

In the pre-training the process is splitted into multiple phases one for train the algorithm on a large dataset but not prepared and cleaned well, and other phase with a smaller one in comparison with the first phase's dataset but well prepared and cleaned carefully in our case - phase 1, 174K samples - phase 2, 90k samples.

After the pre-training phase, the model should be able to predict the next token (next-token prediction, auto-regressive models) to test the language and the knoweldge precision.

### Fine-tuning

In the fine-tuning phase we could give the model the ability to not only predict the next token, you could give it the ability to answer a question, summarize, translate, etc.

### Steps for exploring the data

1. Load the datasets with the specific splits.
2. Explore the columns and for each split (for both pretrain and finetune) within the dataset explore the following: number of rows, null values, language distribution, character length distribution, Approximate token length per MODEL SAMPLE (CHAR_PER_TOKEN, CONTEXT_LENGTHS) and the distribution based on the Context length (what will be dropped and what will remain within each one of the context lengths 512, 1024, 2048).

## Phase 2 - Building the Tokenizer

### Cleaning & Normalization

All the cleaning has been done within the file called cleaning.py

### Build BPE Tokenizer

Problem BPE solves: Word-level tokens (some words may not appear with the corpus, so the inference this be handled as unknown word), Character-level tokens (characters carry little meaning, and sequences will be very very long).

Note: Python is slow while dealing with sequences like the tokenizer (doing multiple steps of processing on top of some text), so using transofmers (built on top of rust) will fill this gap between the python and the tokenizer processing steps, it uses something called **Python bindings** in a library called **pyo3** in **Rust**, this will give the ability to write the functions in Python code but it will be executed behind the scenes using **Rust**.

