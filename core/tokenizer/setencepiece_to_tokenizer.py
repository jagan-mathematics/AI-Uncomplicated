import argparse
import os
import sentencepiece as spm
import regex as re
import json

import core.utils.sentencepiece_model_pb2 as model_loader


UNSUSED_PIECE_PATTERM = "<unused\d+>"

def get_unused_peices(model):
    unused_peices = []
    peices = [model.id_to_piece(id) for id in range(model.get_piece_size())]
    for peice in peices:
        print(peice)
        if re.match(UNSUSED_PIECE_PATTERM, peice):
            unused_peices.append(peice)
    return unused_peices, peices


def convert(model_path, special_tokens, save_path):
    name = os.path.basename(model_path).rsplit(".", 1)[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = spm.SentencePieceProcessor(model_file=model_path)
    unused_slots, peices = get_unused_peices(model)
    available_unused_slots = len(unused_slots) - 1
    if len(special_tokens) > available_unused_slots:
        raise ValueError(f"Avialble special token slots {available_unused_slots} but given {len(special_tokens)}")

    m = model_loader.ModelProto()
    m.ParseFromString(open(model_path, 'rb').read())

    special_token_map = {}
    for piece in special_tokens:
        if available_unused_slots > -1:
            special_token_map[f"<unused{available_unused_slots}>"] = piece
            available_unused_slots -= 1

    fillable_peices = list(special_token_map.keys())
    for p in m.pieces:
        if p.piece in fillable_peices:
            p.piece = special_token_map[p.piece]


    save_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"spm_buffer.model"), 'wb') as f:
        f.write(m.SerializeToString())

    with open(os.path.join(save_path, f"tokenizer.config"), "w") as f:
        config = {
            "vocab_size": model.vocab_size(),
            "available_unused_slots": available_unused_slots,
            "special_tokens": special_tokens,
        }
        json.dump(config, f)

def main(args):
    convert(args.model_path, args.special_tokens, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--special_tokens", nargs='+', default=[])
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    main(args)

