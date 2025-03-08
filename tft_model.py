import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 定义一个简单的时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_length],
                self.data[index+self.seq_length])

# 定义一个简单的注意力机制
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # 计算注意力权重
        weights = self.attention_weights(lstm_out).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        # 计算加权和
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

# 定义TFT模型
class TFTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TFTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SimpleAttention(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        predictions = self.linear(context)
        return predictions

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    plt.plot(outputs.detach().numpy(), label='predictions')
    plt.plot(targets.numpy(), label='targets')
    plt.legend()
    plt.show()

def generate_sample_data(length=100):
    """
    生成有规律的样本数据，较为复杂的函数
    """
    data = torch.zeros(length, 1)
    for i in range(length):
        data[i] = 0.2*torch.sin(torch.tensor(i*0.1)) + i/100
    return data

# 主函数
if __name__ == "__main__":
    # 示例数据
    data = generate_sample_data()
    plt.plot(data)
    plt.show()

    seq_length = 10
    dataset = TimeSeriesDataset(data, seq_length)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = TFTModel(input_size=1, hidden_size=32, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=20) 

    # 测试模型