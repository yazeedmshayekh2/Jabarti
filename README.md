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

## Phase 2.1 - Building the Tokenizer

### Cleaning & Normalization

All the cleaning has been done within the file called cleaning.py

Normalizing the text, Norm title, etc. any other required cleaning step, these depends on your data and your use case, how you want to deal with the data, what it will be used for.

### Build BPE Tokenizer

Problem BPE solves: Word-level tokens (some words may not appear with the corpus, so the inference this be handled as unknown word), Character-level tokens (characters carry little meaning, and sequences will be very very long).

Note: Python is slow while dealing with sequences like the tokenizer (doing multiple steps of processing on top of some text), so using transofmers (built on top of rust) will fill this gap between the python and the tokenizer processing steps, it uses something called **Python bindings** in a library called **pyo3** in **Rust**, this will give the ability to write the functions in Python code but it will be executed behind the scenes using **Rust**.

In the file called `build_tokenizer.py`, we've trained the tokenizer on the corpus after cleaning it using the cleaning steps built in the file called `cleaning.py`. The components of the tokenizer were:

1. The Normalizer.
2. Pre-tokenizer ("Metaspace"), here you could find the difference between the types of the pre-tokenizers:

## Pre-tokenizer Comparison

How different pre-tokenizers (Hugging Face `tokenizers`) split the same sentence:

**Input:** `Hello world!! It's 2026`

| Pre-tokenizer | Used by | Output pieces | Behavior |
|---|---|---|---|
| `WhitespaceSplit` | Simplest baseline | `Hello` · `world!!` · `It's` · `2026` | Splits on spaces only. Punctuation stays glued to words. |
| `Whitespace` | Regex `\w+\|[^\w\s]+` | `Hello` · `world` · `!!` · `It` · `'` · `s` · `2026` | Separates words from punctuation, but a run of punctuation stays together. |
| `BertPreTokenizer` | BERT, WordPiece | `Hello` · `world` · `!` · `!` · `It` · `'` · `s` · `2026` | Like `Whitespace`, but every punctuation character becomes its own piece. |
| `ByteLevel` | GPT-2, RoBERTa, Llama 3 | `Hello` · `Ġworld` · `!!` · `ĠIt` · `'s` · `Ġ2026` | Space is kept as `Ġ` on the next word, so text is perfectly reversible. Works on bytes: no unknown tokens ever. |
| `Metaspace` | SentencePiece, T5, Llama 2 | `▁Hello` · `▁world!!` · `▁It's` · `▁2026` | Replaces spaces with `▁` and splits there. |
| `Punctuation` | Usually inside a `Sequence` | `Hello world` · `!` · `!` · ` It` · `'` · `s 2026` | Only isolates punctuation; ignores spaces entirely. |
| `Digits` | `individual_digits=True` | `Hello world!! It's ` · `2` · `0` · `2` · `6` | Only splits numbers, digit by digit. Helps models do arithmetic consistently. |
| `Sequence` | Combines several | `Hello` · `world!!` · `It's` · `2` · `0` · `2` · `6` | Example: `[WhitespaceSplit, Digits]`. Each step runs on the output of the previous one. |

### Key takeaways

- **Whitespace-only** (`WhitespaceSplit`, `Metaspace`) keeps punctuation attached to words.
- **Punctuation-aware** (`Whitespace`, `BertPreTokenizer`) separates punctuation; BERT splits every character.
- **Byte-level** (`ByteLevel`) is lossless and never produces unknown tokens.
- **Digits** splitting gives consistent number handling; combine it with others via `Sequence`.

---

3. Decoder, it could be as same as the pre-tokenizer (Metaspace) or it could be different, like if you want to deal with different type of languages as input(English) and output(Arabic).

---

## Phase 2.2 - Building the Tokenizer

After we've built and trained our own tokenizer from scratch and downloaded the `tokenizer.json` file, we're gonna now to use it within the LLM tokenizer in file `tokenizer.py` within the folder `jabart_llm`. The steps would be as follows, loading the tokenizer after training it, normalize, encode, tokenize, and decode.

## Phase 3 - Embedding

All the upcoming layer within the transformer architecture would be built using the `Pytorch`, because it's well-suited for building neural networks, it has three features:

1. It has a way to store the values (Tensor).
2. It has a way to work with the CUDA and GPU cores easily.
3. It has a way to calculate the gradients easily without you even notice that.

---

In this phase we've built the embedding layers (Both token and positional embeddings), you can find it within the file called `embeddings.py`, and you can find the tests within the file `jabart_llm/assets/notebooks/tokenizer.ipynb`.

## Phase 4 - Attention
