import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def loss_function(params, data):
    k_ac, k1, k2, alpha = params
    total_loss = 0
    
    # 遍历数据集
    for i in range(len(data) - 1):
        # 获取当前时刻数据
        T, T_evap, f_comp, O_valve, v_fan, T_out, Q_int = data[i]
        # 获取下一时刻真实值
        T_next, T_evap_next = data[i+1][0], data[i+1][1]
        
        # 计算预测值
        dT_dt = (k_ac * (T_evap - T) - U_wall * A_wall * (T_out - T) + Q_int) / (rho * V * Cp)
        dT_evap_dt = (-k1 * (f_comp/f_max) * (1 - np.exp(-k2 * O_valve)) * (v_fan/v_max) 
                      + alpha * (T - T_evap))
        
        # 计算预测的下一时刻值
        T_pred = T + dT_dt * dt
        T_evap_pred = T_evap + dT_evap_dt * dt
        
        # 累加损失
        total_loss += (T_next - T_pred)**2 + (T_evap_next - T_evap_pred)**2
    
    return total_loss / (len(data) - 1)

# 使用scipy.optimize.minimize进行参数优化

# 初始参数猜测
initial_params = [1.0, 1.0, 1.0, 0.1]  # [k_ac, k1, k2, alpha]

# 优化参数
result = minimize(
    loss_function, 
    initial_params, 
    args=(train_data,),
    method='L-BFGS-B', 
    bounds=[(0, None)]*4
)

# 获取优化后的参数
optimized_params = result.x
k_ac_opt, k1_opt, k2_opt, alpha_opt = optimized_params

# 在测试集上验证模型
test_loss = loss_function(optimized_params, test_data)
print(f"测试集损失: {test_loss:.4f}")

# 可视化预测结果
def simulate_model(params, initial_conditions, data):
    T, T_evap = initial_conditions
    predictions = []
    
    for i in range(len(data)):
        # 获取当前时刻输入
        f_comp, O_valve, v_fan, T_out, Q_int = data[i]
        
        # 计算微分
        dT_dt = (k_ac_opt * (T_evap - T) - U_wall * A_wall * (T_out - T) + Q_int) / (rho * V * Cp)
        dT_evap_dt = (-k1_opt * (f_comp/f_max) * (1 - np.exp(-k2_opt * O_valve)) * (v_fan/v_max) 
                      + alpha_opt * (T - T_evap))
        
        # 更新状态
        T += dT_dt * dt
        T_evap += dT_evap_dt * dt
        predictions.append((T, T_evap))
    
    return np.array(predictions)

# 绘制预测结果
predictions = simulate_model(optimized_params, (T0, T_evap0), test_data)
plt.plot(test_data[:, 0], label='真实房间温度')
plt.plot(predictions[:, 0], label='预测房间温度')
plt.legend()
plt.show()