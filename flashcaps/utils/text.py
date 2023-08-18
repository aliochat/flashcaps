from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import (
    NFKC,
    Lowercase,
    Strip,
    Sequence as NormalizerSequence,
)
from tokenizers.pre_tokenizers import (
    Punctuation,
    Metaspace,
    Sequence as PreTokenizerSequence,
)
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
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = NormalizerSequence([NFKC(), Lowercase(), Strip()])
    tokenizer.pre_tokenizer = PreTokenizerSequence([Punctuation(), Metaspace()])
    trainer = WordLevelTrainer(
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        show_progress=True,
    )

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $0 [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)]
    )
    tokenizer.enable_padding(
        pad_id=0, pad_token="<PAD>", pad_type="post", pad_to_multiple_of=8
    )
    tokenizer.decoder = MetaspaceDecoder(add_prefix_space=True)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer


def get_special_token(token_name: str) -> str:
    """
    Get the special token with the given name.
    The following special tokens are available: padding_token, bos_token, eos_token, unk_token.

    Parameters
    ----------
    token_name : str
        Name of the special token. Use 'all' to return the entire dictionary of special tokens.

    Returns
    -------
    str or dict
        The special token string or the entire dictionary of special tokens if 'all' is specified.


    """
    special_tokens = {
        "padding_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
    }

    if token_name == "all":
        return special_tokens

    elif token_name in special_tokens:
        return special_tokens[token_name]

    else:
        raise ValueError(
            f"Token name '{token_name}' is not in the list of available special tokens. Available special tokens: {', '.join(special_tokens.keys())}. Use 'all' to return the entire dictionary of special tokens."
        )
