from tokenizers import pre_tokenizers
from typing import List
import regex as re
from tokenizers import NormalizedString, PreTokenizedString, Regex, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import BPE
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import normalizers

from tokenizers import ByteLevelBPETokenizer, decoders, processors


class CustomNormalizer:
    def normalize(self, normalized: NormalizedString):
        # Most of these can be replaced by a `Sequence` combining some provided Normalizer,
        # (ie Sequence([ NFKC(), Replace(Regex("\s+"), " "), Lowercase() ])
        # and it should be the prefered way. That being said, here is an example of the kind
        # of things that can be done here:
        normalized.nfkc()
        normalized.filter(lambda char: not char.isnumeric())
        normalized.replace(Regex("\s+"), " ")
        normalized.replace(Regex("\.+"), ".")

tokenizer = ByteLevelBPETokenizer(add_prefix_space=False, unicode_normalizer="nfkc")
tokenizer.normalizer = Normalizer.custom(CustomNormalizer())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Digits(individual_digits=True),
    # PreTokenizer.custom(CustomPreTokenizer()),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
])

tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()


import glob

files = glob.glob("data_points/*.txt")

FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"

tokenizer.train(
    files = files,
    vocab_size=50000,
    min_frequency=5,
    show_progress=True,
    special_tokens=[
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<|start_of_turn|>",
        "<|end_of_turn|>",
        "<|start_system_prompt|>",
        "<|end_system_prompt|>",
        FIM_PREFIX,
        FIM_MIDDLE,
        FIM_SUFFIX
    ]
)



tokenizer.save_model("trained_tokenizer/", "gpt_tokenizer_50k")

# Create Dummy normalizer for saving
tokenizer.normalizer = normalizers.NFKC()
tokenizer.save("trained_tokenizer/tokenizer.json")
