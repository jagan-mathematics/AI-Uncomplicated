import regex as re
import sentencepiece as spm
import torch
import json
import os

class SPMTokenizer:
    def __init__(self, tokenizer_path, bos_peice="<s>", eos_peice="</s>", padding_peice="<pad>"):
        self.path = tokenizer_path
        self.model = spm.SentencePieceProcessor(model_file=os.path.join(tokenizer_path, "spm_buffer.model"))

        with open(os.path.join(tokenizer_path, "tokenizer.config"), "r") as handler:
            self.config = json.load(handler)

        self.peices = [self.model.id_to_piece(id) for id in range(self.model.get_piece_size())]
        self.unused_token_pattern = "<unused\d+>"

        self.token_slots = self.config["available_unused_slots"]
        self.filled_slots = 0

        self.special_tokens = self.config["special_tokens"]
        self.__special_token_slot_map = {}

        self.start_token_idx = self.model.PieceToId(bos_peice)
        self.end_token_idx = self.model.PieceToId(eos_peice)
        self.pad_token_idx = self.model.PieceToId(padding_peice)

    # def add_special_token(self, token_value):
    #     if self.filled_slots < self.token_slots:
    #         self.special_tokens.append(token_value)
    #         self.__special_token_slot_map[f"<unused{self.filled_slots}>"] = token_value
    #         self.filled_slots += 1
    #     raise ValueError("Slot full")


    @property
    def available_special_token_slots(self):
        return self.token_slots


    def get_unused_peices(self):
        unused_peices = []
        for peice in self.peices:
            if re.match(self.unused_token_pattern, peice):
                unused_peices.append(peice)
        return unused_peices

    @property
    def vocab_size(self):
        return self.model.vocab_size()

    @vocab_size.setter
    def vocab_size(self, value):
        raise ValueError("Assigning value not supported")

    def encode(self, inputs, add_special_tokens=True, return_type="pt"):
        if isinstance(inputs, str):
            inputs = [inputs]
        encoded_tokens = self.model.Encode(inputs)
        if add_special_tokens:
            for idx in range(len(encoded_tokens)):
                encoded_tokens[idx] = [self.start_token_idx] + encoded_tokens[idx] + [self.end_token_idx]

        if return_type is None:
            return {"input_ids": encoded_tokens, "attention_mask": None}
        elif return_type == "pt":
            paddded_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in encoded_tokens], batch_first=True, padding_value=self.pad_token_idx).long()
            attention_mask = (paddded_tokens != self.pad_token_idx).to(torch.int32)
            return {"input_ids": paddded_tokens, "attention_maks": attention_mask}
        else:
            raise ValueError("unsupported return type")

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy().tolist()
        return self.model.Decode(tokens)

    # def save_to_folder(self, folder_path, name="model"):
    #     m = model_loader.ModelProto()
    #     m.ParseFromString(open(self.path , 'rb').read())

    #     fillable_peices = list(self.__special_token_slot_map.keys())
    #     for p in m.pieces:
    #         if p.piece in fillable_peices:
    #             p.piece = self.__special_token_slot_map[p.piece]


    #     with open(os.path.join(folder_path, f"spm_buffer.model"), 'wb') as f:
    #         f.write(m.SerializeToString())

    #     with open(os.path.join(folder_path, f"tokenizer.config"), "w") as f:
    #         config = {
    #             "vocab_size": self.vocab_size,
    #             "available_unused_slots": self.available_special_token_slots,
    #             "special_tokens": self.special_tokens,
    #         }
    #         json.dump(config, f)