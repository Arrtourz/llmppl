import os
import logging
from time import sleep
import numpy as np
import tiktoken
import openai
from typing import List, Tuple, Optional

class GPTLogProb:
    """Language Model with encoding handling for log probability calculation."""

    def __init__(self, model: str, sleep_time: int = 10):
        logging.basicConfig(level=logging.INFO)
        logging.info(f'Initializing model: `{model}`')
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = self.api_key
        self.model = model
        self.sleep_time = sleep_time
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_logprobs(self, input_text: str) -> Tuple[List[List[Tuple[str, float]]], List[str]]:
        token_indices = self.encoding.encode(input_text)
        tokens = [self.encoding.decode([i]) for i in token_indices]

        all_logprobs = []

        for i in range(0, len(tokens)):
            current_prompt = "".join(tokens[:i])
            
            while True:
                try:
                    completion = openai.Completion.create(
                        engine=self.model,
                        prompt=current_prompt,
                        logprobs=10,
                        max_tokens=1,
                        temperature=0
                    )
                    break
                except openai.error.RateLimitError:
                    logging.info(f'Rate limit exceeded. Waiting for {self.sleep_time} seconds.')
                    sleep(self.sleep_time)
                except Exception as e:
                    logging.error(f'OpenAI API error: {e}')
                    raise

            if completion.choices[0].logprobs.top_logprobs:
                logprobs = completion.choices[0].logprobs.top_logprobs[0]
                all_logprobs.append(list(logprobs.items()))
            else:
                all_logprobs.append([])

        return all_logprobs, tokens

def calculate_ppl(input_text: str, model_name: str = "gpt-3.5-turbo-instruct") -> float:
    scorer = GPTLogProb(model=model_name)
    logprobs, tokens = scorer.get_logprobs(input_text)
    
    log_sum = 0.0
    token_count = 0

    print("Token-wise log probabilities:")
    for idx, token_set in enumerate(logprobs):
        actual_token = tokens[idx]
        token_dict = dict(token_set)
        
        if actual_token in token_dict:
            token_logprob = token_dict[actual_token]
            log_sum += token_logprob
            token_count += 1
            print(f"Token: '{actual_token}', LogProb: {token_logprob:.4f}")
        else:
            print(f"Token: '{actual_token}', LogProb: N/A (not in top 10 predictions)")
            logging.warning(f"Token '{actual_token}' not in top 10 predictions for position {idx}")

    if token_count == 0:
        raise ValueError("No valid tokens found for perplexity calculation")

    average_log_prob = log_sum / token_count
    perplexity = np.exp(-average_log_prob)
    
    # print(f"\nTotal tokens: {token_count}")
    # print(f"Average log probability: {average_log_prob:.4f}")
    
    return perplexity

if __name__ == "__main__":
    input_text = 'The quick brown fox jumps over the lazy dog.'
    
    perplexity = calculate_ppl(input_text)
    print(f"\nPerplexity: {perplexity:.4f}")