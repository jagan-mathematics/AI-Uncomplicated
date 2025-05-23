import contextlib
import json
import functools
from transformers import AutoTokenizer
import numpy as np
from jinja2 import Template
import numpy as np
from dataclasses import field

from dataclasses import dataclass
from typing import Dict, Optional, Any, TypedDict

class JSONLState(TypedDict):
    position: int
    file_path: str
    current_iter: int


class TokenizerState(TypedDict):
    jsonl_state: Any
    sequence_length: int
    gen_seq_length: int
    add_system_prompt: bool
    add_template: bool
    add_spl_token: bool
    n_views: int
    rng_state: Dict[str, Any]
    tokenizer_path: Optional[str]


class BatchingState(TypedDict):
    tokenizer_state: Any
    sequence_index: int
    batch_size: int
    prefetch_size: int
    rng_state: Dict[str, Any]


class InputProcessorState(TypedDict):
    jsonl_state: Any
    add_system_prompt: bool




def jsonl_iterator(file_path, position, current_iter):
    with open(file_path, "r") as file:
        file.seek(position)
        while line := file.readline():
            state = JSONLState(
                file_path=file_path,
                position=file.tell(),
                current_iter=current_iter
            )
            yield json.loads(line), state


def loop_on_jsonl(
    file_path: str,
    position: int,
    current_iter: int
):
    """Makes the block jsonl iterator infinite and updates n_iter counter"""
    try:
        while True:
            it = jsonl_iterator(file_path, position, current_iter)
            for content, jsonl_state in it:
                yield content, jsonl_state
            current_iter += 1
            position = 0
    finally:
        it.close()


@functools.lru_cache(maxsize=None)
def get_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)



def construct_input_prompt(table_list):
    table_templates = []
    for table in table_list:
        if "table_type" not in table or table["table_type"].lower() != "table":
            continue
        table_name = table["table_name"]
        columns_template = ""
        constraints_template = ""
        columns_template_list = []
        constraint_template_list = []

        for column in table["columns"]:
            columns_template = f"name: {column['name']}\ntype: {column['type']}"
            if "default" in column:
                columns_template += f"\ndefault: {column['default']}"

            if "constraints" in column:
                constraints = '\n- '.join(column['constraints']).strip('\n')
                columns_template += f"\nconstraints: {constraints}"
            columns_template = "\n".join(["\t" + line for line in columns_template.split("\n")])
            columns_template_list.append(columns_template)

        for constraint in table["constraints"]:
            constraints_template = f"columns: {constraint['columns']}\constraint_type: {constraint['constraint_type']}"
            if "references" in constraint:
                ref_cols = '\n-'.join(constraint['columns'])
                constraints_template += f"\nreferences: \ntable: {constraint['table'] if 'table' in constraint else None}\ncolumns:{ref_cols}"

            if "expression" in constraint:
                constraints = '\n- '.join(constraint['expression']).strip('\n')
                constraints_template += f"\nconstraints: {constraints}"
            constraints_template = "\n".join(["\t" + line for line in constraints_template.split("\n")])
            constraint_template_list.append(constraints_template)

        col = '\n\n'.join(columns_template_list)
        const = '\n\n'.join(constraint_template_list)
        table_templates.append(f"""
Name: {table_name}
Columns:
{col}
Constrinats:
{const}
""")
    return "Tables:\n" + "\n\n".join(table_templates)


def tokenizer(
    data_iterator,
    tokenizer_path,
    sequence_length,
    gen_seq_length,
    add_system_prompt,
    add_template,
    add_spl_token,
    n_views,
    rng_state):

    system_prompt = """You are an SQL Expert your Task is to find the related table and colunns which are need to anwer the query, the output should be in list of table json format:"""

    token_encoder = get_tokenizer(tokenizer_path)

    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    for data_point, jsonl_state in data_iterator:
        input_ = construct_input_prompt(json.loads(data_point["sql_table_schema"]))
        if add_system_prompt:
            input_ = f"{system_prompt}\n\n{input_}\n\n##Query: {data_point['prompt']}"
        else:
            input_ = f"{input_}\n\n##Query: {data_point['prompt']}"

        if add_template:
            input_ = f"<|im_start|>user\n{input_}<|im_end|><|im_start|>assistant\n"
        else:
            input_ = f"<|im_start|>\n{input_}\n"

        labels_ = f"```json\n{data_point['sql_schema_linkage']}\n```<|im_end|>"

        tokens = token_encoder.encode(input_, padding=False, truncation=False, max_length=sequence_length - gen_seq_length, add_special_tokens=add_spl_token)
        labels = token_encoder.encode(labels_,  padding=False, truncation=False, max_length=gen_seq_length, add_special_tokens=add_spl_token)

        label_start_idx = len(tokens) - 1

        assert len(tokens) + len(labels) < sequence_length

        input_tokens = np.concatenate([tokens, labels])
        input_tokens = np.lib.stride_tricks.sliding_window_view(
            input_tokens, n_views, axis=0
        )
        input_tokens = np.array(input_tokens)
        input_tokens[:label_start_idx, 1] = -100

        pad_rows = sequence_length - input_tokens.shape[0]

        input_tokens = np.pad(input_tokens, ((0, pad_rows), (0, 0)), mode='constant', constant_values=token_encoder.pad_token_type_id)
        input_tokens[input_tokens[:, 1] == token_encoder.pad_token_type_id, 1] = -100

        yield input_tokens, TokenizerState(
            jsonl_state=jsonl_state,
            sequence_length=sequence_length,
            gen_seq_length=gen_seq_length,
            add_system_prompt=add_system_prompt,
            add_template=add_template,
            add_spl_token=add_spl_token,
            n_views=n_views,
            rng_state=rng.bit_generator.state,
            tokenizer_path=tokenizer_path
        )


def batch_shuffle(data_iterator, batch_size,  seq_len, n_views, prefetch_size, state):
    prefetch_buffer = -1 * np.ones(
        (prefetch_size * batch_size, seq_len, n_views), dtype=int
    )

    sequence_idx = state["sequence_index"]

    rng_state = state["rng_state"]
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    iteration_state = state["tokenizer_state"]

    assert (
        sequence_idx >= 0 and sequence_idx < prefetch_size
    ), "Prefetch state sequence_idx should be in 0 <= sequence_idx < prefetch_size."

    for i in range(prefetch_size * batch_size):
        data, next_it_state = next(data_iterator)
        prefetch_buffer[i] = data

    rng.shuffle(prefetch_buffer, axis=0)


    for i in range(sequence_idx*batch_size):
        data, _ = next(data_iterator)
        prefetch_buffer[i] = data


    idx = sequence_idx

    while True:
        if idx == prefetch_size - 1:
            iteration_state = next_it_state
            rng_state = rng.bit_generator.state

        state = BatchingState(
            tokenizer_state=iteration_state,
            sequence_index=(idx + 1) % prefetch_size,
            rng_state=rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size
        )

        yield prefetch_buffer[idx * batch_size : (idx + 1) * batch_size].copy(), state

        for i in range(batch_size):
            prefetch_buffer[idx * batch_size + i], iteration_state = next(data_iterator)


        if idx == prefetch_size - 1:
            next_it_state = iteration_state
            rng.shuffle(prefetch_buffer, axis=0)

        idx = (idx + 1) % prefetch_size



@dataclass
class TokenizerArgs:
    tokenizer_name: str
    add_system_prompt: str
    add_template: bool
    add_special_tokens: bool


@dataclass
class DataArgs:
    data_path: str
    batch_size: int = 2
    seq_len: int = 2048
    gen_seq_len: int = 256
    n_views: int = 2
    seed: int = 42
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)


def build_state(
    data_path: str,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    gen_seq_len: int,
    n_views: int,
    seed: int,
    tokenizer_name: str,
    add_system_prompt: bool,
    add_template: bool,
    add_special_tokens: bool,
):
    tokenizer_rng_state = np.random.default_rng(
        (seed)
    ).bit_generator.state

    jsonl_state = JSONLState(position=0,
                             file_path=data_path,
                             current_iter=0)

    tokenizer_state = TokenizerState(
        jsonl_state=jsonl_state,
        tokenizer_path=tokenizer_name,
        sequence_length=seq_len,
        n_views=n_views,
        gen_seq_length=gen_seq_len,
        add_system_prompt=add_system_prompt,
        add_template=add_template,
        add_spl_token=add_special_tokens,
        rng_state=tokenizer_rng_state
    )

    batch_shuffle_state = np.random.default_rng(
        (seed + 1)
    ).bit_generator.state

    batch_shuffle_state = BatchingState(
        tokenizer_state=tokenizer_state,
        sequence_index=0,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        rng_state=batch_shuffle_state
    )

    return batch_shuffle_state


@contextlib.contextmanager
def build_dataloader(
    state: BatchingState
):
    tokenizer_state = state["tokenizer_state"]
    jsonl_state = tokenizer_state["jsonl_state"]

    data_iterator = loop_on_jsonl(
        jsonl_state["file_path"],
        jsonl_state["position"],
        current_iter=jsonl_state["current_iter"]
    )

    tokenized_iterator = tokenizer(
        data_iterator=data_iterator,
        tokenizer_path=tokenizer_state["tokenizer_path"],
        sequence_length=tokenizer_state["sequence_length"],
        gen_seq_length=tokenizer_state["gen_seq_length"],
        add_system_prompt=tokenizer_state["add_system_prompt"],
        add_template= tokenizer_state["add_template"],
        add_spl_token= tokenizer_state["add_spl_token"],
        n_views = tokenizer_state["n_views"],
        rng_state=tokenizer_state["rng_state"]
    )

    data_it = batch_shuffle(
        data_iterator=tokenized_iterator,
        batch_size=state["batch_size"],
        seq_len=tokenizer_state["sequence_length"],
        n_views=tokenizer_state["n_views"],
        prefetch_size=state["prefetch_size"],
        state=state
    )

    yield data_it
    data_iterator.close()
    tokenized_iterator.close()
    data_it.close()

def build_data_loader_from_state(state):
    return functools.partial(build_dataloader, state)()


def init_dataloader_state_from_args(
        args: DataArgs,
):
    return build_state(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        gen_seq_len=args.gen_seq_len,
        n_views=args.n_views,
        seed=args.seed,
        tokenizer_name=args.tokenizer.name,
        add_system_prompt=args.tokenizer.add_system_prompt,
        add_template=args.tokenizer.add_template,
        add_special_tokens=args.tokenizer.add_special_tokens
    )
