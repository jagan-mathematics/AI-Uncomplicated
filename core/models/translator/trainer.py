from core.models.translator.config import ModelConfig, DatasetConfig, TrainingConfig
from core.dataloaders.dataloader import load_tokenizer

## Initialize configurations
model_config = ModelConfig(model_name="Construe",
                           num_layers = 2,
                            padding_id = 0,
                            hidden_dim = 512,
                            intermediate_dim = 3072,
                            max_positions = 2048,
                            layer_norm_eps = 1e-05,
                            model_max_sequence = 2048,
                            num_heads = 8,
                            attention_dropout = 0.1)

dataset_config = DatasetConfig(dataset_path="./dataset",
                               dataset_shuffle=True)
training_config = TrainingConfig(tokenizer_path="/root/AI-Uncomplicated/core/tokenizer/bpe/pre_trained/europian_ml")

## Load model and tokenizer

tokenizer =  load_tokenizer(training_config.tokenizer_path)


