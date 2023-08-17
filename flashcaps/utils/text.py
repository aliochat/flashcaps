from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Strip, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Punctuation, Metaspace, Sequence as PreTokenizerSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.trainers import WordLevelTrainer
from typing import Iterable

def create_word_level_tokenizer(corpus: Iterable[str], min_frequency: int) -> Tokenizer:
    """
    Create and train a word-level tokenizer on the given corpus.

    Parameters
    ----------
    corpus : iterable of str
        The corpus to train the tokenizer on.
    min_frequency : int
        The minimum frequency a token must have to be included in the vocabulary.

    Returns
    -------
    tokenizer : Tokenizer
        The trained tokenizer.

    Example
    -------
    >>> corpus = ["This is a sentence.", "This is another sentence."]
    >>> tokenizer = create_word_level_tokenizer(corpus, min_frequency=1)
    
    # Encoding a sequence
    >>> output = tokenizer.encode("This is a sentence.")
    >>> print(output.tokens)
    ['[BOS]', 'this', 'is', 'a', 'sentence', '.', '[EOS]']
    >>> print(output.ids)
    [1, 2, 3, 4, 5, 6, 2]

    # Decoding a sequence
    >>> decoded_output = tokenizer.decode(output.ids)
    >>> print(decoded_output)
    'this is a sentence.'
    """
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.normalizer = NormalizerSequence([NFKC(), Lowercase(), Strip()])
    tokenizer.pre_tokenizer = PreTokenizerSequence([Punctuation(), Metaspace()])
    trainer = WordLevelTrainer(
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        show_progress=True
        )
    
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $0 [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    tokenizer.enable_padding(pad_id=0, pad_token="<PAD>", pad_type="post", pad_to_multiple_of=8)
    tokenizer.decoder = MetaspaceDecoder(add_prefix_space=True)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer
