import llmppl

def test_mlm_ppl():
    """
    Test Masked Language Model Perplexity (MLM) using BERT model.
    """
    mlm = llmppl.MLMPPL(model_name='bert-base-uncased')
    text = "The quick brown fox jumps over the lazy dog."
    ppl = mlm.calculate_ppl(text)
    assert ppl > 0


def test_decoder_ppl():
    """
    Test Decoder Perplexity (Causal LM) using GPT-2 model.
    """
    decoder = llmppl.DecoderPPL(model_name='gpt2')
    text = "The quick brown fox jumps over the lazy dog."
    ppl = decoder.calculate_ppl(text)
    assert ppl > 0


def test_encoder_decoder_ppl():
    """
    Test Encoder-Decoder Perplexity using T5 model.
    """
    enc_dec = llmppl.EncoderDecoderPPL(model_name='t5-small')
    input_text = "translate English to German: The quick brown fox jumps over the lazy dog."
    output_text = "Der schnelle braune Fuchs springt über den faulen Hund."
    ppl = enc_dec.calculate_ppl(input_text, output_text)
    assert ppl > 0


def test_llmppl_mlm():
    """
    Test LLMPPL static method for MLM model.
    """
    text = "The quick brown fox jumps over the lazy dog."
    ppl = llmppl.LLMPPL.get_perplexity(text, model_type='mlm')
    assert ppl > 0


def test_llmppl_decoder():
    """
    Test LLMPPL static method for Decoder model.
    """
    text = "The quick brown fox jumps over the lazy dog."
    ppl = llmppl.LLMPPL.get_perplexity(text, model_type='clm', model_name='gpt2')
    assert ppl > 0


def test_llmppl_encoder_decoder():
    """
    Test LLMPPL static method for Encoder-Decoder model.
    """
    input_text = "translate English to German: The quick brown fox jumps over the lazy dog."
    output_text = "Der schnelle braune Fuchs springt über den faulen Hund."
    ppl = llmppl.LLMPPL.get_perplexity(input_text, output_text, model_type='enc-dec', model_name='t5-small')
    assert ppl > 0

def test_gpt35_logprob():
    model = llmppl.GPTLogProb(model='gpt-3.5-turbo-instruct')
    text = "The quick brown fox jumps over the lazy dog."
    logprobs, tokens = model.get_logprobs(text)
    assert len(logprobs) > 0
    assert len(tokens) > 0

def test_llama2_ppl():
    llama2 = llmppl.Llama2PPL()
    text = "The quick brown fox jumps over the lazy dog."
    ppl = llama2.calculate_ppl(text)
    assert ppl > 0

def test_rwkv_ppl():
    rwkv = llmppl.RWKVPPL()
    text = "The quick brown fox jumps over the lazy dog."
    ppl = rwkv.calculate_ppl(text)
    assert ppl > 0

def test_mamba_ppl():
    mamba = llmppl.MambaPPL()
    text = "The quick brown fox jumps over the lazy dog."
    ppl = mamba.calculate_ppl(text)
    assert ppl > 0

def test_mixtral_ppl():
    mixtral = llmppl.MixtralPPL()
    text = "The quick brown fox jumps over the lazy dog."
    ppl = mixtral.calculate_ppl(text)
    assert ppl > 0
