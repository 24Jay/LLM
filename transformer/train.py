import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import model_config
from dataset import BilingualDataset, make_causal_mask
from mini_transformer import build_transformer
from tqdm import tqdm


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
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_word.item()).type_as(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()

    count = 0

    source_text = []
    expected = []
    predicted = []

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
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)

            source_text.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    # if writer:



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

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)

    writer = SummaryWriter(config["experiment_name"])
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)

    model_file_name  = model_config.latest_weights_file_path(config)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    global_step = 0
    for epoch in range(config["num_epochs"]):
        torch.cuda.empty_cache()

        model.train()

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
            writer.flush()

            loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1



    model_file_name = config.get_weights_file_path(config, epoch)
    torch.save({
        "epoch":epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict()
    }, model_file_name)


    




if __name__ == "__main__":

    config = model_config.get_config()

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


