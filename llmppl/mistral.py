import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import math

class MixtralPPL:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        primary_gpu = 0 if self.device == 'cuda' else None
        max_memory = {
            0: "16GB",
            "cpu": "48GB"
        } if primary_gpu is not None else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="sequential",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder="offload_folder",
        )
        
        self.model.eval()

    def calculate_ppl(self, text: str) -> float:
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                ppl = math.exp(outputs.loss.item())
            
            torch.cuda.empty_cache()
            return ppl
            
        except Exception as e:
            return float('inf')

def calculate_ppl(text: str, model_name: str = "mistralai/Mathstral-7b-v0.1") -> float:
    model = MixtralPPL(model_name)
    return model.calculate_ppl(text)

if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a subfield of artificial intelligence and linguistics.",
        "Mixtral is a large language model developed by Mistral AI using sparse mixture of experts.",
        "The concept of perplexity in language models quantifies how well a probability model predicts a sample."
    ]

    model = MixtralPPL()
    for text in texts:
        ppl = model.calculate_ppl(text)
        print(f"Perplexity: {ppl:.4f}")