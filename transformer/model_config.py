from pathlib import Path


def get_config(experiment_name):
    base_path = f"./transformer/{experiment_name}"

    dataset_path = Path(f"{base_path}/dataset")
    tokenizer_path = Path(f"{base_path}/tokenizer")
    model_path = Path(f"{base_path}/model")
    log_path = Path(f"{base_path}/logs")
    
    dataset_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)


    return {
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 1e-4,
        "seq_len": 200,
        "d_model": 512,
        # "datasource": "opus_books",
        "datasource": "Helsinki-NLP/opus-100",
        "src_lang": "en",
        "tgt_lang": "zh",
        "num_samples": 20000,
        "zh_vocab_size": 10000,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "dataset_path": dataset_path,
        "log_path": log_path,
        "experiment_name": base_path,
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


# def get_exp_path(config):
#     return config["experiment_name"]

# def get_model_path(config):
#     return str(Path(".") / config["experiment_name"] / "model")

# def get_log_path(config):
#     return str(Path(".") / config["experiment_name"] / "log")