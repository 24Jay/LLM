import torch
import torch.nn as nn


def my_cross_entropy_loss(logits, labels):
    """
    计算交叉熵损失
    """
    # 计算对数概率

    # log_probs = torch.log_softmax(logits, dim=-1)
    # logit 2 softmax probability
    probs = torch.softmax(logits, dim=-1)

    # probs 2 log_probs
    log_probs = torch.log(probs)
    # 计算交叉熵损失

    # labels 2 one_hot
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1))

    # print( logits, probs, log_probs, labels, torch.sum(log_probs * one_hot_labels, dim=-1) )

    loss = torch.sum(log_probs * one_hot_labels, dim=-1)
    print(f"sum loss: {loss.shape}")
    loss = -torch.mean(torch.sum(log_probs * one_hot_labels, dim=-1))
    print(f"mean loss: {loss.shape}")
    return loss

# 定义不同的类别数量
num_classes_list = [2, 5, 10, 100, 1000, 10000, 20000]
# num_classes_list = [2]
num_samples = 1

for num_classes in num_classes_list:
    print("-" * 30)

    correct_labels = torch.zeros(num_samples, dtype=torch.long)

    # 生成完全正确的预测 logits
    correct_logits = torch.zeros(num_samples, num_classes)
    correct_logits[:, 0] = 10  # 确保第一个类别概率接近 1
    print(f"correct_labels: {correct_labels}")

    # 生成完全错误的预测 logits
    wrong_logits = torch.zeros(num_samples, num_classes)
    wrong_logits[:, 1] = 10  # 确保第二个类别概率接近 1

    # 生成随机预测
    random_logits = torch.randn(num_samples, num_classes)

    criterion = nn.CrossEntropyLoss()

    # 计算最小损失
    min_loss = criterion(correct_logits, correct_labels)
    # 计算最大损失
    max_loss = criterion(wrong_logits, correct_labels)
    # 计算随机损失
    random_loss = criterion(random_logits, correct_labels)

    print(f"类别数量: {num_classes}")
    print(f"最小损失: {min_loss.item()}, my loss: {my_cross_entropy_loss(correct_logits, correct_labels)}")
    print(f"最大损失: {max_loss.item()}, 理论最大损失: {torch.log(torch.tensor(num_classes)).item()}, my loss: {my_cross_entropy_loss(wrong_logits, correct_labels)}")
    print(f"随机损失: {random_loss.item()}, my loss: {my_cross_entropy_loss(random_logits, correct_labels)}")
    print("-" * 30)
    print("\n")