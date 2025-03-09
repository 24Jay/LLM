import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_estimation(func, a, b, num_samples=10000):
    """使用蒙特卡洛方法估计函数在区间[a, b]上的积分值

    参数:
        func: 要积分的函数
        a: 积分下限
        b: 积分上限
        num_samples: 采样点数量，默认为10000

    返回:
        积分估计值, 采样点x, 采样点y
    """
    x_samples = np.random.uniform(a, b, num_samples)
    y_samples = func(x_samples)
    integral = (b - a) * np.mean(y_samples)
    
    return integral, x_samples, y_samples


def test_function(x):
    """测试函数，包含三角函数和指数函数
    
    参数:
        x: 输入值
        
    返回:
        函数值加上随机噪声
    """
    y = np.sin(x) + np.cos(x) + 0.5 * np.exp(-1 * x)
    noise = np.random.normal(0, 0.1)
    
    return y + noise


def direct_sampling(func, a, b, num_samples=10000):
    """直接采样
    
    参数:
        func: 要采样的函数
        a: 采样区间下限
        b: 采样区间上限
        num_samples: 采样点数量，默认为10000

    返回:
        采样点x, 采样点y
    """
    # 直接采样需要从目标分布中生成样本
    # 这里假设func是目标概率密度函数
    # 首先在区间[a,b]上均匀采样候选点
    candidates = np.random.uniform(a, b, num_samples)

    # 计算每个候选点的概率密度
    probs = func(candidates)
    
    # 对概率密度进行归一化
    probs /= np.sum(probs)
    
    # 根据概率密度进行采样
    indices = np.random.choice(len(candidates), size=num_samples, p=probs)
    x_samples = candidates[indices]
    
    # 计算对应的函数值
    y_samples = func(x_samples)
    
    return x_samples, y_samples


# 什么是拒绝采样
# 拒绝采样是一种蒙特卡洛方法，用于从复杂分布中采样。它通过在目标分布附近构建一个简单的提议分布，并拒绝不符合目标分布的样本，从而得到符合目标分布的样本。

# 拒绝采样的步骤
# 1. 选择一个简单的提议分布，例如均匀分布
# 2. 在提议分布上生成候选点
# 3. 计算候选点在目标分布下的概率密度
# 4. 根据概率密度进行采样
def rejection_sampling(func, a, b, num_samples=10000):
    """拒绝采样
    
    参数:
        func: 要采样的函数
        a: 采样区间下限
        b: 采样区间上限

    """
    # 提议分布选择均匀分布
    # 在区间[a,b]上均匀采样候选点
    candidates = np.random.uniform(a, b, num_samples)

    # 计算每个候选点的概率密度
    probs = func(candidates)

    # 计算提议分布的概率密度
    # proposal_probs = np.ones(candidates, 1.0 / (b-a))  # 均匀分布的概率密度函数
    proposal_probs = np.full_like(candidates, 1.0 / b-a)  # 均匀分布的概率密度函数
    print(proposal_probs)

    # 计算接受概率
    acceptance_probs = probs / proposal_probs

    # 根据接受概率进行采样
    accepted = np.random.uniform(0, 1, num_samples) < acceptance_probs

    # 返回接受样本
    return candidates[accepted]


def test_rejection_sampling():
    """测试拒绝采样"""
    # 定义目标分布
    def target_distribution(x):
        # 对函数进行归一化处理
        Z = np.sqrt(np.pi)  # 正态分布的归一化常数
        return np.exp(-x**2) / Z
    
    def exponential_distribution(x, lam=1.0):
        """指数分布函数
        
        参数:
            x: 输入值
            lam: 指数分布的参数λ，默认为1.0
            
        返回:
            概率密度值
        """
        # 指数分布本身已经是归一化的概率密度函数，无需额外归一化
        return np.where(x >= 0, lam * np.exp(-lam * x), 0)
    
    def complex_distribution(x):
        """定义一个复杂的混合分布函数
        
        参数:
            x: 输入值
            
        返回:
            概率密度值
        """
        # 包含多个高斯分布的混合
        gaussian1 = 0.3 * np.exp(-0.5 * ((x - 1) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
        gaussian2 = 0.5 * np.exp(-0.5 * ((x + 1) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))
        gaussian3 = 0.2 * np.exp(-0.5 * ((x - 3) / 0.3)**2) / (0.3 * np.sqrt(2 * np.pi))
        
        # 添加一个指数分布成分
        exponential = 0.1 * np.where(x >= 0, 0.5 * np.exp(-0.5 * x), 0)
        
        # 返回混合分布
        return gaussian1 + gaussian2 + gaussian3 + exponential
    
    # 使用拒绝采样方法从目标分布中采样
    # 调整采样参数，增加采样点数量并缩小采样区间
    # samples = rejection_sampling(target_distribution, -3, 3, num_samples=100000)
    samples = rejection_sampling(exponential_distribution, -3, 5, num_samples=100000)
    samples = rejection_sampling(complex_distribution, -3, 5, num_samples=100000)

    # 绘制采样结果
    plt.figure(figsize=(10, 6))
    
    # 绘制理论分布曲线
    x = np.linspace(-3, 10, 1000)
    y = complex_distribution(x)
    plt.plot(x, y, label='Target Distribution', linewidth=2)
    
    # 绘制采样直方图
    plt.hist(samples, bins=100, density=True, alpha=0.6, 
             label='Rejection Sampling')
    
    # 添加图形元素
    plt.title('Rejection Sampling Test')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()


def importance_sampling(target_dist, proposal_dist, proposal_sampler, num_samples=10000):
    """重要性采样方法
    
    参数:
        target_dist: 目标分布函数
        proposal_dist: 建议分布函数
        proposal_sampler: 从建议分布中采样的函数
        num_samples: 采样数量
        
    返回:
        采样点, 权重
    """
    # 从建议分布中采样
    samples = proposal_sampler(num_samples)
    
    # 计算重要性权重
    weights = target_dist(samples) / proposal_dist(samples)
    
    # 归一化权重
    weights /= np.sum(weights)
    
    return samples, weights

def test_importance_sampling():
    # 定义目标分布（标准正态分布）
    target_dist = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
    # 定义复杂目标分布
    def complex_target_dist(x):
        # 包含多个高斯分布的混合
        gaussian1 = 0.3 * np.exp(-0.5 * ((x - 1) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
        gaussian2 = 0.5 * np.exp(-0.5 * ((x + 1) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))
        gaussian3 = 0.2 * np.exp(-0.5 * ((x - 3) / 0.3)**2) / (0.3 * np.sqrt(2 * np.pi))
        
        # 添加一个指数分布成分
        exponential = 0.1 * np.where(x >= 0, 0.5 * np.exp(-0.5 * x), 0)
        
        # 返回混合分布
        return gaussian1 + gaussian2 + gaussian3 + exponential
    
    target_dist = complex_target_dist
    
    # 定义建议分布（均匀分布）
    proposal_dist = lambda x: np.where((-3 <= x) & (x <= 3), 1/6, 0)
    proposal_sampler = lambda n: np.random.uniform(-3, 3, n)
    
    # 执行重要性采样
    samples, weights = importance_sampling(target_dist, proposal_dist, proposal_sampler)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
    # 绘制目标分布
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, target_dist(x), label='Target Distribution', linewidth=2)
    
    # 绘制加权直方图
    plt.hist(samples, bins=50, weights=weights, density=True, 
             alpha=0.6, label='Importance Sampling')
    plt.hist(samples, bins=50, density=True, alpha=0.3, label='Raw Samples')
    
    plt.title('Importance Sampling Test')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()







def test_monte_carlo_estimation():
    # 执行蒙特卡洛估计
    res, x_samples, y_samples = monte_carlo_estimation(test_function, -1, 1)

    # 绘制函数图像和采样点
    x = np.linspace(-1, 1, 100)
    y = test_function(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Test Function")
    plt.scatter(x_samples, y_samples, color='red', 
                label='Monte Carlo Samples', alpha=0.5)
    plt.title("Monte Carlo Integration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_rejection_sampling()
    test_importance_sampling()