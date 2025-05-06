Transformer from scratch

> ![Coding a Transformer from scratch on PyTorch](https://www.youtube.com/watch?v=ISNdQcPhsts)

[Transformer from scratch](https://github.com/karpathy/minGPT)

## 1. transale_en_zh

```python
{
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

```

epoch: 42
cross_entropy_loss: 1.7
bleu: 0.18
问题: 训练过程中 bleu 指标在下降

image.png
