import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import math
from huggingface_hub import HfFolder
import os
import importlib.metadata as metadata  # 替代 pkg_resources

def check_dependencies():
    required = {'torch', 'transformers', 'protobuf', 'bitsandbytes', 'accelerate'}
    installed = {pkg for pkg in metadata.distributions()}
    installed_names = {pkg.metadata['Name'] for pkg in installed}
    missing = required - installed_names

    if missing:
        raise ImportError(f"Missing required packages: {', '.join(missing)}. Please install them using pip.")

class Llama2PPL:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        token = HfFolder.get_token() or os.getenv('HUGGING_FACE_TOKEN')
        if not token:
            raise ValueError("No Hugging Face token found. Please log in using `huggingface-cli login` or set the HUGGING_FACE_TOKEN environment variable.")

        print("Loading tokenizer...")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, token=token)
        
        print("Loading model...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            token=token
        )
        self.model.eval()

        # Check for older GPU drivers
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            print("Note: If you're using an RTX 4000 series GPU, please ensure your GPU drivers are up to date to avoid P2P issues.")

    def calculate_ppl(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            
        loss = outputs.loss
        ppl = math.exp(loss.item())
        return ppl

def main():
    try:
        check_dependencies()
        llama_ppl = Llama2PPL()

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a subfield of artificial intelligence and linguistics.",
            "Llama 2 is a large language model developed by Meta AI.",
            "It's a point he's made many times before, and it's one that he'"
        ]

        for text in texts:
            ppl = llama_ppl.calculate_ppl(text)
            print(f"Text: {text}")
            print(f"Perplexity: {ppl:.4f}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
