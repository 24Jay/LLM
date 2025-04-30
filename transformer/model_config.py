from pathlib import Path


def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        # "datasource": "Helsinki-NLP/opus-100",
        "src_lang": "en",
        "tgt_lang": "zh",
        "num_examples": 30000,
        "model_folder": "weights",
        "model_basename": "transformer_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "./transformer/translate_en_zh",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"./transformer/{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
# def latest_weights_file_path(config):
#     model_folder = f"./transformer/{config['model_folder']}"

#     model_filename = f"{config['model_basename']}*"
#     weights_files = list(Path(model_folder).glob(model_filename))
#     if len(weights_files) == 0:
#         return None
#     weights_files.sort()
#     return str(weights_files[-1])


def get_exp_path(config):
    return config["experiment_name"]

def get_model_path(config):
    return str(Path(".") / config["experiment_name"] / "model")

def get_log_path(config):
    return str(Path(".") / config["experiment_name"] / "log")