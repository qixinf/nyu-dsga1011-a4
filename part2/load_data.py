from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast

PAD_IDX = 0

@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    # Keep multi-character SQLite operators intact during SentencePiece decoding.
    tokenizer.add_tokens(["<", "<=", "<>"])
    return tokenizer


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.inputs, self.targets = self.process_data(data_folder, split, get_tokenizer())

    def process_data(self, data_folder, split, tokenizer):
        folder = Path(data_folder)
        questions = load_lines(folder / f"{split}.nl")
        inputs = tokenizer(questions, max_length=128, truncation=True).input_ids
        if split == "test":
            return inputs, None
        queries = load_lines(folder / f"{split}.sql")
        if len(queries) != len(questions):
            raise ValueError(f"Mismatched question and SQL counts for {split}")
        targets = tokenizer(queries, max_length=512, truncation=True).input_ids
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        target = None if self.targets is None else torch.tensor(self.targets[idx])
        return torch.tensor(self.inputs[idx]), target
    
def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask, initial_decoder_inputs = test_collate_fn(batch)
    decoder_targets = pad_sequence([target for _, target in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.cat((initial_decoder_inputs, decoder_targets[:, :-1]), dim=1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([source for source, _ in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX)
    # Pretrained T5 uses its pad token as the decoder start token.
    initial_decoder_inputs = encoder_ids.new_full((len(batch), 1), PAD_IDX)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x

def token_statistics(lines, tokenizer):
    """Count text tokens before padding, truncation, or model-specific special tokens."""
    encoded = tokenizer(lines, add_special_tokens=False, padding=False,
                        truncation=False, return_attention_mask=False)
    return sequence_statistics(encoded["input_ids"])


def sequence_statistics(sequences):
    vocabulary = {token for sequence in sequences for token in sequence}
    return sum(map(len, sequences)) / len(sequences), len(vocabulary)


def print_q4_statistics():
    data_dir = Path(__file__).resolve().parent / "data"
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    columns = []
    processed_columns = []
    for split in ("train", "dev"):
        questions = load_lines(data_dir / f"{split}.nl")
        queries = load_lines(data_dir / f"{split}.sql")
        if not questions or len(questions) != len(queries):
            raise ValueError(f"Expected nonempty paired NL and SQL files for {split}")
        nl_mean, nl_vocab = token_statistics(questions, tokenizer)
        sql_mean, sql_vocab = token_statistics(queries, tokenizer)
        columns.append((len(questions), f"{nl_mean:.2f}", f"{sql_mean:.2f}", nl_vocab, sql_vocab))

        dataset = T5Dataset(data_dir, split)
        # The final token is EOS; padding is added later by the collate function.
        nl_mean, nl_vocab = sequence_statistics([ids[:-1] for ids in dataset.inputs])
        sql_mean, sql_vocab = sequence_statistics([ids[:-1] for ids in dataset.targets])
        processed_columns.append((f"{nl_mean:.2f}", f"{sql_mean:.2f}", nl_vocab, sql_vocab))

    labels = ("Number of examples", "Mean sentence length", "Mean SQL query length",
              "Vocabulary size (natural language)", "Vocabulary size (SQL)")
    rows = list(zip(labels, *columns))
    processed_rows = list(zip(labels[1:], *processed_columns))
    for title, table_rows in (("Table 1: Before preprocessing", rows),
                              ("Table 2: After preprocessing (SQL operator tokens)", processed_rows)):
        print(title)
        print("| Statistic | Train | Dev |")
        print("|---|---:|---:|")
        for label, train, dev in table_rows:
            print(f"| {label} | {train} | {dev} |")
        print()


if __name__ == "__main__":
    print_q4_statistics()
    raise SystemExit(0)