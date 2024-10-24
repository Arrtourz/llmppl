import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from typing import Union, List

class MambaPPL:
    def __init__(self, model_name: str = "state-spaces/mamba-130m-hf", device: str = None):
        """
        Initialize Mamba model for perplexity calculation.
        
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
        print(f"Model loaded successfully: {model_name}")

    def calculate_ppl(self, text: Union[str, List[str]], verbose: bool = False) -> Union[float, List[float]]:
        """
        Calculate perplexity for given text(s).
        
        Args:
            text: Input text or list of texts
            verbose: Whether to print token-level information
            
        Returns:
            Perplexity score(s)
        """
        if isinstance(text, str):
            return self._calculate_single_ppl(text, verbose)
        return [self._calculate_single_ppl(t, verbose) for t in text]

    def _calculate_single_ppl(self, text: str, verbose: bool = False) -> float:
        """Calculate perplexity for a single text."""
        try:
            # Tokenize text
            encodings = self.tokenizer(text, return_tensors='pt').to(self.device)
            input_ids = encodings.input_ids
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

            if verbose:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                print(f"\nTokens: {tokens}")
                print(f"Number of tokens: {len(tokens)}")
                print(f"Loss: {loss:.4f}")
            
            # Calculate perplexity
            ppl = math.exp(loss)
            return ppl
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            raise

def main():
    try:
        # Available model sizes
        MODEL_SIZES = {
            'tiny': "state-spaces/mamba-130m-hf",
            'small': "state-spaces/mamba-370m-hf",
            'base': "state-spaces/mamba-790m-hf",
            'large': "state-spaces/mamba-2.8b-hf"
        }

        # Initialize model (using smallest variant)
        print("\nInitializing Mamba model...")
        mamba_ppl = MambaPPL(model_name=MODEL_SIZES['tiny'])

        # Test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a subfield of artificial intelligence and linguistics.",
            "Mamba is a state space model that provides an efficient alternative to Transformers.",
            # Add more test texts here
        ]

        # Individual text processing with verbose output
        print("\nProcessing individual texts:")
        for text in texts:
            ppl = mamba_ppl.calculate_ppl(text, verbose=True)
            print(f"Text: {text}")
            print(f"Perplexity: {ppl:.4f}\n")

        # Batch processing example
        print("\nBatch processing example:")
        ppls = mamba_ppl.calculate_ppl(texts)
        for text, ppl in zip(texts, ppls):
            print(f"Text: {text[:50]}...")
            print(f"Perplexity: {ppl:.4f}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()