import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import math
from tqdm import tqdm
import numpy as np

class MLMPPL:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def calculate_ppl(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]

        nll = []
        for text in tqdm(texts, desc="Processing texts"):
            encodings = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            nll_sum = 0
            token_count = 0

            for i in range(input_ids.size(1)):
                masked_input_ids = input_ids.clone()
                masked_input_ids[:, i] = self.tokenizer.mask_token_id

                with torch.no_grad():
                    outputs = self.model(masked_input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                probs = torch.softmax(logits, dim=-1)
                true_token_probs = probs[0, i, input_ids[0, i]]
                nll_sum -= torch.log(true_token_probs).item()
                token_count += 1

            nll.append(nll_sum / token_count)

        ppl = [math.exp(x) for x in nll]
        return ppl if len(ppl) > 1 else ppl[0]

class DecoderPPL:
    def __init__(self, model_name='gpt2', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def calculate_ppl(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        ppls = []
        for text in tqdm(texts, desc="Processing texts"):
            encodings = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            max_length = self.model.config.max_position_embeddings
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
            ppls.append(ppl.item())

        return ppls if len(ppls) > 1 else ppls[0]

class EncoderDecoderPPL:
    def __init__(self, model_name='t5-small', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def calculate_ppl(self, input_texts, output_texts):
        if isinstance(input_texts, str):
            input_texts = [input_texts]
            output_texts = [output_texts]
        
        ppls = []
        for input_text, output_text in tqdm(zip(input_texts, output_texts), desc="Processing texts"):
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=self.tokenizer(output_text, return_tensors="pt")["input_ids"].to(self.device))
            
            loss = outputs.loss
            ppl = math.exp(loss.item())
            ppls.append(ppl)

        return ppls if len(ppls) > 1 else ppls[0]

class LLMPPL:
    @staticmethod
    def get_perplexity(input_text, output_text=None, model_name='bert-base-uncased', model_type='mlm'):
        if model_type == 'mlm':
            calculator = MLMPPL(model_name)
            return calculator.calculate_ppl(input_text)
        elif model_type == 'clm':
            calculator = DecoderPPL(model_name)
            return calculator.calculate_ppl(input_text)
        elif model_type == 'enc-dec':
            if output_text is None:
                raise ValueError("output_text must be provided for encoder-decoder models")
            calculator = EncoderDecoderPPL(model_name)
            return calculator.calculate_ppl(input_text, output_text)
        else:
            raise ValueError("model_type must be 'mlm', 'clm', or 'enc-dec'")

# Example usage
if __name__ == "__main__":
    texts = [
        "Natural language processing is a subfield of linguistics and artificial intelligence.",
        "The quick brown fox jumps over the lazy dog."
    ]

    print("\nTesting Encoder-Decoder models:")
    enc_dec_models = ['t5-small', 'google/flan-t5-small']
    input_texts = [
        "translate English to German: The quick brown fox jumps over the lazy dog.",
        "summarize: Natural language processing is a subfield of linguistics and artificial intelligence."
    ]
    output_texts = [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "NLP is a field combining linguistics and AI."
    ]
    for model in enc_dec_models:
        print(f"\nCalculating perplexity using {model}:")
        ppls = LLMPPL.get_perplexity(input_texts, output_texts, model, 'enc-dec')
        for input_text, output_text, ppl in zip(input_texts, output_texts, ppls):
            print(f"Input: {input_text}")
            print(f"Output: {output_text}")
            print(f"Perplexity: {ppl:.4f}\n")

    # Example with a longer text
    longer_input = """
    Translate the following text to French:
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers to 
    process and analyze large amounts of natural language data.
    """
    longer_output = """
    Le traitement du langage naturel (TLN) est un sous-domaine de la linguistique, de l'informatique et de l'intelligence artificielle
    qui s'intéresse aux interactions entre les ordinateurs et le langage humain, en particulier à la manière de programmer les ordinateurs pour
    traiter et analyser de grandes quantités de données en langage naturel.
    """
    print("\nLonger input text:")
    print(longer_input)
    print("\nExpected output:")
    print(longer_output)
    print("\nCalculating perplexity using T5:")
    ppl_longer = LLMPPL.get_perplexity(longer_input, longer_output, 't5-small', 'enc-dec')
    print(f"Perplexity: {ppl_longer:.4f}")