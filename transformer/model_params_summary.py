from train import get_model, get_ds
import model_config
from torchsummary import summary
import torch


# summary(model, (1, 512))


def compute_model_size(model):
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return trainable_params, total_params


if __name__ == "__main__":
    config = model_config.get_config()


    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab()))

    state = torch.load("./transformer/translate_en_zh/model/transformer_10.pth")
    model.load_state_dict(state["model_state_dict"])

    trainable_params, total_params = compute_model_size(model)