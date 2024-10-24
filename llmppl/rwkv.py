import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

class RWKVPPL:
    def __init__(self, model_name="RWKV/rwkv-raven-7b", device=None):
        """
        Initialize RWKV model for PPL calculation.
        
        Args:
            model_name (str): Name of the model on HuggingFace
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully")

    def calculate_ppl(self, text: str) -> float:
        """
        Calculate perplexity for the given text.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Perplexity score
        """
        try:
            # Tokenize text
            encodings = self.tokenizer(text, return_tensors='pt', add_special_tokens=True).to(self.device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss.item()
            
            # Calculate perplexity
            ppl = math.exp(loss)
            return ppl
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            raise

def main():
    try:
        print("Initializing RWKV model...")

        # Use the RWKV/rwkv-raven-7b model
        rwkv_ppl = RWKVPPL(model_name="RWKV/rwkv-raven-7b")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a subfield of artificial intelligence and linguistics.",
            "RWKV is a linear attention model that combines the best of RNN and Transformer.",
        ]

        print("\nCalculating perplexity for test texts:")
        for text in texts:
            try:
                ppl = rwkv_ppl.calculate_ppl(text)
                print(f"\nText: {text}")
                print(f"Perplexity: {ppl:.4f}")
            except Exception as e:
                print(f"Error processing text: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have enough GPU memory and all required packages are installed.")

if __name__ == "__main__":
    main()
