
# llmppl

`llmppl` is a Python package for calculating text perplexity using various language models, including GPT-3.5, Llama 2, RWKV, and Mixtral.

## Installation

You can install this package via pip:

```bash
pip install llmppl
```

## Usage

Here are some examples of how to use `llmppl` to calculate text perplexity.

### GPT-3.5 Turbo
export OPENAI_API_KEY='YOUROPENAIAPIKEY'

```python
from llmppl import GPTLogProb

model = GPTLogProb(model='gpt-3.5-turbo-instruct')
text = "The quick brown fox jumps over the lazy dog."
logprobs, tokens = model.get_logprobs(text)
print("Log probabilities:", logprobs)
```

### Llama 2

```python
from llmppl import Llama2PPL

llama = Llama2PPL(model_name="meta-llama/Llama-2-7b-hf")
text = "The quick brown fox jumps over the lazy dog."
ppl = llama.calculate_ppl(text)
print(f"Perplexity: {ppl}")
```

### RWKV

```python
from llmppl import RWKVPPL

rwkv = RWKVPPL(model_name="RWKV/rwkv-raven-7b")
text = "The quick brown fox jumps over the lazy dog."
ppl = rwkv.calculate_ppl(text)
print(f"Perplexity: {ppl}")
```

## Perplexity Calculation for Core Models

The `llmppl` package supports perplexity (PPL) calculation using various language model types including Masked Language Models (MLM), Causal Language Models (CLM), and Encoder-Decoder models. Here are examples of how to use these models for perplexity calculation.

### Masked Language Model (MLM) Perplexity

```python
from llmppl import MLMPPL

mlm = MLMPPL(model_name='bert-base-uncased')
text = "The quick brown fox jumps over the lazy dog."
ppl = mlm.calculate_ppl(text)
print(f"Perplexity (MLM): {ppl}")
```

### Causal Language Model (CLM) Perplexity

```python
from llmppl import DecoderPPL

decoder = DecoderPPL(model_name='gpt2')
text = "The quick brown fox jumps over the lazy dog."
ppl = decoder.calculate_ppl(text)
print(f"Perplexity (CLM): {ppl}")
```

### Encoder-Decoder Model Perplexity

```python
from llmppl import EncoderDecoderPPL

enc_dec = EncoderDecoderPPL(model_name='t5-small')
input_text = "translate English to German: The quick brown fox jumps over the lazy dog."
output_text = "Der schnelle braune Fuchs springt über den faulen Hund."
ppl = enc_dec.calculate_ppl(input_text, output_text)
print(f"Perplexity (Encoder-Decoder): {ppl}")
```

### General Perplexity Calculation via LLMPPL

```python
from llmppl import LLMPPL

# Masked Language Model (MLM)
text = "The quick brown fox jumps over the lazy dog."
ppl = LLMPPL.get_perplexity(text, model_type='mlm', model_name='bert-base-uncased')
print(f"Perplexity (MLM): {ppl}")

# Causal Language Model (CLM)
ppl = LLMPPL.get_perplexity(text, model_type='clm', model_name='gpt2')
print(f"Perplexity (CLM): {ppl}")

# Encoder-Decoder Model
input_text = "translate English to German: The quick brown fox jumps over the lazy dog."
output_text = "Der schnelle braune Fuchs springt über den faulen Hund."
ppl = LLMPPL.get_perplexity(input_text, output_text, model_type='enc-dec', model_name='t5-small')
print(f"Perplexity (Encoder-Decoder): {ppl}")
```

## Dependencies

This project requires the following Python packages:

- `torch==2.5.0`
- `transformers==4.45.2`
- `openai==0.28.0`
- `tqdm==4.66.5`
- `sentencepiece==0.2.0`
- `bitsandbytes==0.44.1`
- `accelerate==1.0.1`
- `protobuf==5.28.2`
- `tiktoken==0.8.0`

## Contributing

Contributions are welcome! If you have ideas or find bugs, feel free to submit an issue or a pull request.

## Citation

If this package is helpful to your work, please consider citing the following paper:

Xu, Z., & Sheng, V. S. (2024, March). Detecting AI-Generated Code Assignments Using Perplexity of Large Language Models. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 38, No. 21, pp. 23155-23162).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
