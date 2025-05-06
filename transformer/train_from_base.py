import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torchmetrics.text import SacreBLEUScore

import os
from pathlib import Path
import model_config
from dataset import BilingualDataset, make_causal_mask
from mini_transformer import build_transformer
from tqdm import tqdm
from datetime import datetime
import chinese_tokenizer
import model_params_summary


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    for _ in range(max_len):
        decoder_mask = make_causal_mask(decoder_input.shape[1]).type_as(source_mask).to(device)

        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_word.item()).type_as(source).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=4):
    model.eval()

    count = 0

    source_texts = []
    expected = []
    predicted, predicted_raw = [], []

    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split("\n")[0].split(" ")
            console_width = int(console_width)
    except:
        console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.shape[0] == 1, "batch size must be 1 for validation task"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_raw = tokenizer_tgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)#.replace(" ", "")
            model_out_text = model_out_text_raw.replace(" ", "")

            source_texts.append(source_text)
            expected.append(target_text)
            predicted_raw.append(model_out_text_raw)
            predicted.append(model_out_text)

            if count <= 5:
                print_msg('-'*console_width)
                print_msg(f"Source: {source_text}")
                print_msg(f"Target: {target_text}")
                print_msg(f"Predicted: {model_out_text}")

            if count == num_examples:
                # print_msg('-'*console_width)
                break
        if writer:
            print(f"validate bleu: {len(predicted)}")
            bleu = SacreBLEUScore(tokenize="zh")
            expected = [[e] for e in expected]
            bleu_score = bleu(predicted, expected)
            writer.add_scalar("bleu", bleu_score, global_step)
            print_msg(f"BLEU score: {bleu_score}")
            writer.flush()
        print_msg('-'*console_width)




def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_chinese_tokenizer(config, ds, lang):
    tokenizer_path = Path(f"{config["tokenizer_path"]}/tokenizer_{lang}.json")

    if not Path.exists(tokenizer_path):
        # 初始化 BPE 模型
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=10000, min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_or_build_tokenizer(config, ds, lang):
    if lang == "zh":
        return get_or_build_chinese_tokenizer(config, ds, lang)

    tokenizer_path = Path(f"{config["tokenizer_path"]}/tokenizer_{lang}.json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size = config["zh_vocab_size"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    if config["tgt_lang"] == "zh":
        ds_raw = chinese_tokenizer.get_zh_en_dataset(config)    
    else:
        ds_raw = load_dataset(config["datasource"],f"{config['src_lang']}-{config['tgt_lang']}", split="train" )

    print(f"==============={config['datasource']}, {config['src_lang']}-{config['tgt_lang']}: len = {len(ds_raw)}===============")
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


def train_model(config, base_model = ""):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"use device: {device}")

    device = torch.device(device)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)

    writer = SummaryWriter(log_dir=f"{config['log_path']}/{datetime.now()}")
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.Step(optim, step_size=1, gamma=0.6)

    initial_epoch = 0
    if len(base_model) > 0 and Path(base_model).exists():
        print(f"Preloading model from {base_model}")
        state = torch.load(base_model)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optim.load_state_dict(state["optimizer_state_dict"])

    else:
        print(f"no model found, start from scratch: {base_model}")

    model_params_summary.compute_model_size(model)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    global_step = 0
    for epoch in range(initial_epoch, initial_epoch + config["num_epochs"]):
        torch.cuda.empty_cache()

        model.train()

        for param_group in optim.param_groups:
            param_group["lr"] = config["lr"]

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_ouput = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_ouput, encoder_mask, decoder_input, decoder_mask)
            projected_output = model.project(decoder_output)

            label = batch["label"].to(device)
            loss = loss_fn(projected_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("Train/loss", loss.item(), global_step=global_step)
            # lr
            writer.add_scalar("Train/lr", optim.param_groups[0]["lr"], global_step=global_step)
            writer.flush()

            loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,config["seq_len"],device, \
                       lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=100)


        torch.save({
            "epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict()
        }, f"{config['model_path']}/transformer_{epoch}.pth")


    




if __name__ == "__main__":

    config = model_config.get_config(experiment_name="translate_en_zh")

    train_model(config, base_model="./transformer/translate_en_zh/model/transformer_20.pth")

    # print(train_model(config, base_model="./transformer/translate_en_zh/model/transformer_19.pth"))


