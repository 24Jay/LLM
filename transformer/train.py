import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config
from dataset import BilingualDataset
from mini_transformer import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(config["datasource"],f"{config['src_lang']}-{config['tgt_lang']}", split="train" )

    # build source and target tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt_lang"])

    # split train and val
    train_ds_size = int(len(ds_raw) * 0.9)
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, len(ds_raw) - train_ds_size]
    )

    train_ds = BilingualDataset( train_ds_raw,  tokenizer_src,   tokenizer_tgt,   config["src_lang"], config["tgt_lang"],config["seq_len"] )
    val_ds = BilingualDataset( val_ds_raw,  tokenizer_src,   tokenizer_tgt,   config["src_lang"], config["tgt_lang"],config["seq_len"] )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"use device: {device}")

    device = torch.device(device)
    




if __name__ == "__main__":

    config = get_config()

    # ds_raw = load_dataset(
    #     config["datasource"],
    #     f"{config['src_lang']}-{config['tgt_lang']}",
    #     split="train",
    # )

    # print(f"data len:", len(ds_raw))

    # max_src, max_tgt = 0, 0
    # for i in range(len(ds_raw)):
    #     max_src = max(max_src, len(ds_raw[i]["translation"][config["src_lang"]]))
    #     max_tgt = max(max_tgt, len(ds_raw[i]["translation"][config["tgt_lang"]]))
    #     # print(f"data {i}: {len(ds_raw[i]["translation"][config['src_lang']])}, {len(ds_raw[i]["translation"][config['tgt_lang']])}")

    # print(f"max src: {max_src}, max tgt: {max_tgt}")
    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=get_config())

    print(train_model(config))


