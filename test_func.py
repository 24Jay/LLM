import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def expansion_valve_function(O, k2):
    return 1 - np.exp(-k2 * O)

# 生成数据
O_values = np.linspace(0, 10, 100)  # O_expansion_valve 的范围
k2 = 0.5  # 常数 k2
y_values = expansion_valve_function(O_values, k2)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(O_values, y_values, label=r'$1 - e^{-k_2 \cdot O_{expansion\ valve}}$')
plt.xlabel(r'$O_{expansion\ valve}$')
plt.ylabel('Output')
plt.title(r'Expansion Valve Function: $1 - e^{-k_2 \cdot O_{expansion\ valve}}$')
plt.grid(True)
plt.legend()
plt.show()