from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from datasets import load_dataset, load_from_disk




def get_zh_en_dataset(num_examples=30000):
    file_path = Path("./ds_file/zh_en_dataset")
    if file_path.exists():
        print("load from local file....")
        ds = load_from_disk(str(file_path))
        print(f"dataset from disk: {len(ds)=}")
        return ds

    # 加载中英文翻译数据集
    ds = load_dataset("wmt/wmt19", "zh-en", split="train")
    ds = ds.filter(lambda example: len(example["translation"]["en"].split()) <= 128)
    ds = ds.filter(lambda example: len(example["translation"]["zh"].split()) <= 128)
    # ds = ds.shuffle(seed=42).select(range(10000))

    # 采样5万条样本
    ds = ds.shuffle(seed=42).select(range(num_examples))

    print(f"wmt/wmt19 data len : {len(ds)}")
    for i in range(5):
        print(ds[i]["translation"])
    
    ds.save_to_disk(str(file_path))

    return ds



get_zh_en_dataset()