from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from datasets import load_dataset, load_from_disk
import model_config
import re


def get_zh_en_dataset(config):
    file_path = Path(f"{config['dataset_path']}/ds_{config["src_lang"]}_{config['tgt_lang']}")
    if file_path.exists():
        print("load from local file....")
        ds = load_from_disk(str(file_path))
        print(f"dataset from disk: {len(ds)=}")
        ds = ds.shuffle(seed=42).select(range(min(config["num_samples"], len(ds))))
        return ds
    else:
        # 加载中英文翻译数据集
        # ds = load_dataset("wmt/wmt19", "zh-en", split="train")
        ds = load_dataset("wmt/wikititles", "zh-en", split="train")
        print(f"wmt/wmt19 data len : {len(ds)}")
        for i in range(5):
            print(i, ": ", ds[i])

        # ds = ds.filter(lambda example: len(example["translation"]["en"].split()) <= 128)

        english_pattern = re.compile(r"[a-zA-Z]")
        # ds = ds.filter(lambda example: len(example["translation"]["zh"].split()) <= 128 and not english_pattern.search(example["translation"]["zh"]))

        # 采样5万条样本
        ds = ds.shuffle(seed=42).select(range(min(config["num_samples"], len(ds))))
    
        
        print(f"sample len: {len(ds)}")
        ds.save_to_disk(str(file_path))

        return ds



get_zh_en_dataset(model_config.get_config("zh_en_base"))