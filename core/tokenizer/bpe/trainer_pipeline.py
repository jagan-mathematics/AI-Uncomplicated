"""Module to Download CC dataset and Train the tokenizer"""
import os
import datasets
import sentencepiece as spm
from typing import List, Iterator, Union, Optional


class Cc100TokenizerTrainer:
    """
    A class to train tokenizers on CC100 datasets for multiple European languages
    """

    def __init__(
            self,
            languages=None,
            base_url: str = "https://data.statmt.org/cc-100/{}.txt.xz",
            output_dir: str = "tokenizer_models",
            vocab_size: int = 64000
    ):
        if languages is None:
            languages = [
                "bg", "hr", "cs", "da", "nl", "en", "et", "fi",
                "fr", "de", "el", "hu", "ga", "it", "lv", "lt",
                "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv"
            ]
        self.languages = languages
        self.base_url = base_url
        self.output_dir = output_dir
        self.vocab_size = vocab_size

        os.makedirs(output_dir, exist_ok=True)

    def _stream_dataset(self, lang: str) -> Iterator[str]:

        download_url = self.base_url.format(lang)
        data = datasets.load_dataset(
            'cc100',
            lang,
            split='train',
            streaming=True
        )

        return (item['text'] for item in data)  # NOTE: STREAMING TEXT DATASET

    def train_multilingual_tokenizer(
            self,
            output_prefix: str = "multilingual_european",
            max_sentences: Optional[int] = 1_000_000
    ) -> None:

        temp_training_file = os.path.join(self.output_dir, f'{output_prefix}_training_data.txt')

        with open(temp_training_file, 'w', encoding='utf-8') as f:
            total_sentences = 0

            for lang in self.languages:
                print(f"Processing language: {lang}")

                lang_stream = self._stream_dataset(lang)

                for text in lang_stream:
                    f.write(text.strip() + '\n')
                    total_sentences += 1

                    if total_sentences >= max_sentences:
                        break

                if total_sentences >= max_sentences:
                    break

        model_path = os.path.join(self.output_dir, output_prefix)
        spm.SentencePieceTrainer.train(
            input=temp_training_file,
            input_format="text",
            model_prefix=model_path,
            model_type='bpe',
            vocab_size=self.vocab_size,
            character_coverage=0.995,
            num_threads=16,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_digits=True,
            treat_whitespace_as_suffix=True,
            byte_fallback=True,
            add_dummy_prefix=True,
            remove_extra_whitespaces=True,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
            max_sentence_length=32000
        )

        print(f"Multilingual Tokenizer trained and saved in {self.output_dir}")

        os.remove(temp_training_file)


trainer = Cc100TokenizerTrainer(
    languages=[
        "en", "fr", "de", "es", "it", "nl",
        "pl", "pt", "sv", "da", "fi"
    ],
    vocab_size=32000
)

trainer.train_multilingual_tokenizer(
    output_prefix="european_multilingual"
)