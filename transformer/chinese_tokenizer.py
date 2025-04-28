from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


# # 使用空白预分词器
# tokenizer.pre_tokenizer = Whitespace()

# # 初始化训练器
# trainer = BpeTrainer(
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
#     vocab_size=30000
# )

# # 训练文件列表 (每行一个句子)
# files = ["train.txt"]

# # 训练分词器
# tokenizer.train(files, trainer)

# # 保存分词器
# tokenizer.save("custom-chinese-tokenizer.json")


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # 初始化 BPE 模型
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=30000, min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer