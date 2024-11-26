from pathlib import Path
from typing import List, Optional, Union
import sentencepiece as spm
import glob
import yaml
import time
import argparse


class CustomTrainingTokenizer:
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 languages: Optional[List[str]] = None,
                 vocab_size: int = 64000,
                 model_type: str = 'bpe',
                 yaml_file_path: Optional[str] = None):
        self.data_path = data_dir
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.languages = languages or ["en", "fr", "de", "es", "it", "nl", "pl", "pt"]
        self.vocab_size = vocab_size
        self.sp_model = None
        if yaml_file_path:
            self.yaml_file_path = yaml_file_path
        self.model_type = model_type

    def get_training_files(self) -> str:
        try:
            path = self.data_path + "**/*.txt"
            training_files = glob.glob(path)

            if not training_files:
                raise ValueError("No training files found!")

            print(f"\nTotal files found: {len(training_files)}")

            return ",".join(training_files)
        except Exception as e:
            print(f"Error collecting training files: {str(e)}")

    def get_yaml_list(self):
        """Read the yaml and get the user fefined symbol and control symbol"""
        try:
            with open(self.yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)

            user_defined_symbols = data.get('user_defined_symbols', [])
            control_symbols = data.get('control_symbols', [])
            return user_defined_symbols, control_symbols
        except Exception as e:
            print(f"Exception While loading yaml file {str(e)}")

    def train_tokenizer(self, model_name: str = "tokenizer", character_coverage: float = 0.995, num_threads: int = 256):
        input_files = self.get_training_files()
        model_prefix = str(self.output_dir / model_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        user_defined_symbols, control_symbols = self.get_yaml_list()

        if user_defined_symbols and control_symbols:
            print(f"========= Training Tokenizer with Given YML File symbols ==========")

        print("Starting tokenizer training...")
        try:
            spm.SentencePieceTrainer.train(
                input=input_files,
                input_format="text",
                model_prefix=model_prefix,
                model_type=self.model_type,
                vocab_size=self.vocab_size,
                character_coverage=character_coverage,
                num_threads=num_threads,
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
                max_sentence_length=64000,
                train_extremely_large_corpus=True,
                control_symbols=control_symbols,
                user_defined_symbols=user_defined_symbols,

            )
            print(f"\nTokenizer trained successfully!")
            print(f"Model saved at: {model_prefix}.model")
            print(f"Vocab saved at: {model_prefix}.vocab")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def load_model(self, model_path: str):
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)
        return self.sp_model

    def encode(self,
               text: Union[str, List[str]],
               out_type: str = str,
               add_bos: bool = True,
               add_eos: bool = True) -> Union[List[str], List[int], List[List[str]], List[List[int]]]:

        if isinstance(text, str):
            return self.sp_model.encode(
                text,
                out_type=out_type,
                add_bos=add_bos,
                add_eos=add_eos
            )
        else:
            return [
                self.sp_model.encode(
                    t,
                    out_type=out_type,
                    add_bos=add_bos,
                    add_eos=add_eos
                ) for t in text
            ]

    def decode(self,
               tokens: Union[List[str], List[int], List[List[str]], List[List[int]]]) -> Union[str, List[str]]:
        if self.sp_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if isinstance(tokens[0], (list, tuple)):
            return [self.sp_model.decode(t) for t in tokens]
        return self.sp_model.decode(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer with custom configurations.")

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the directory containing training data (text files)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the directory where the tokenizer model and vocab will be saved."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000,
        help="Vocabulary size for the tokenizer (default: 32000)."
    )
    parser.add_argument(
        "--yaml_file_path", type=str, default=None,
        help="Path to the YAML file for user-defined and control symbols (optional)."
    )
    parser.add_argument(
        "--model_name", type=str, default="multilingual_tokenizer",
        help="Name prefix for the tokenizer model files (default: tokenizer)."
    )
    parser.add_argument(
        "--model_type", type=str, default="bpe", choices=["bpe", "unigram", "word", "char"],
        help="Type of SentencePiece model (default: bpe)."
    )
    parser.add_argument(
        "--character_coverage", type=float, default=0.995,
        help="Character coverage for training (default: 0.995)."
    )
    parser.add_argument(
        "--num_threads", type=int, default=256,
        help="Number of threads to use during training (default: 256)."
    )
    parser.add_argument(
        "--languages", type=list, default=['en'],
        help="List of Languages to be processed"
    )

    args = parser.parse_args()

    start_time = time.time()
    print("==== TOKENIZER TRAINING STARTED =====")

    tokenizer = CustomTrainingTokenizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        yaml_file_path=args.yaml_file_path,
        model_type=args.model_type,
        languages=args.languages
    )

    tokenizer.train_tokenizer(model_name=args.model_name, character_coverage=args.character_coverage,
                              num_threads=args.num_threads)

    print("Time taken to complete :: ", round(time.time() - start_time, 2))
    print("==== TOKENIZER TRAINED SUCCESSFULLY =====")
