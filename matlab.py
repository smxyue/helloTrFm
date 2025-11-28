import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List
import pickle
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False    # Fix negative sign issue

# =============================== 辅助函数 ===============================

def wrap_to_2pi(angle):
    """将角度包装到[0, 2π]区间 (支持数组)"""
    angle = angle % (2 * np.pi)
    angle[angle < 0] += 2 * np.pi
    return angle

def wrap_to_2pi_scalar(angle):
    """将角度包装到[0, 2π]区间 (标量版本)"""
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return angle

def auv_3dof_euler_2input(xx, tao, dt):
    """原始AUV模型函数"""
    # 状态提取
    x, y, psi, u, v, r = xx
    
    # 控制输入
    tau_u, tau_v, tau_r = tao
    
    # 惯性参数
    m11 = 31.41
    m22 = 65.98
    m66 = 8.33
    
    # 阻尼项
    d11 = 13.5 + 6.68 * abs(u)
    d22 = 66.6 + 196.26 * abs(v)
    d66 = 6.87 + 24.13 * abs(r)
    
    # 动力学模型
    du = (m22 * v * r - d11 * u + tau_u) / m11
    dv = (-m11 * u * r - d22 * v + tau_v) / m22
    dr = ((m11 - m22) * u * v - d66 * r + tau_r) / m66
    
    # 运动学模型
    dx = u * np.cos(psi) - v * np.sin(psi)
    dy = u * np.sin(psi) + v * np.cos(psi)
    dpsi = r
    
    # 前向欧拉积分
    x_next = x + dt * dx
    y_next = y + dt * dy
    psi_next = psi + dt * dpsi
    u_next = u + dt * du
    v_next = v + dt * dv
    r_next = r + dt * dr
    
    return np.array([x_next, y_next, psi_next, u_next, v_next, r_next])

def auv_3dof_euler_2input_limited(xx, tao, dt, u_lim, v_lim, r_lim):
    """带限幅的AUV模型函数"""
    xx_next = auv_3dof_euler_2input(xx, tao, dt)
    
    # 应用限幅
    xx_next[2] = wrap_to_2pi_scalar(xx_next[2])  # 航向角限幅
    xx_next[3] = np.clip(xx_next[3], u_lim[0], u_lim[1])  # 纵向速度限幅
    xx_next[4] = np.clip(xx_next[4], v_lim[0], v_lim[1])  # 横向速度限幅
    xx_next[5] = np.clip(xx_next[5], r_lim[0], r_lim[1])  # 回转角速度限幅
    
    return xx_next

# ===================== 数据生成辅助函数实现 =====================

def generate_constant_single(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim):
    """生成单独恒定输入序列"""
    u_values = np.arange(tau_u_lim[0], tau_u_lim[1] + 10, 10)
    v_values = np.arange(tau_v_lim[0], tau_v_lim[1] + 10, 10)
    
    steps_per_input = total_steps // (len(u_values) + len(v_values))
    tao_seq = np.zeros((3, total_steps))
    current_idx = 0
    
    for i in range(len(u_values)):
        if current_idx >= total_steps:
            break
        end_idx = min(current_idx + steps_per_input, total_steps)
        tao_seq[0, current_idx:end_idx] = u_values[i]
        current_idx = end_idx
    
    for i in range(len(v_values)):
        if current_idx >= total_steps:
            break
        end_idx = min(current_idx + steps_per_input, total_steps)
        tao_seq[1, current_idx:end_idx] = v_values[i]
        current_idx = end_idx
    
    return tao_seq

def generate_combined_rotation(total_steps, dt, tau_u_lim, tau_r_lim):
    """纵向+回转组合输入"""
    u_values = np.arange(tau_u_lim[0], tau_u_lim[1] + 10, 10)
    r_values = np.arange(tau_r_lim[0], tau_r_lim[1] + 10, 10)
    
    steps_per_combination = total_steps // (len(u_values) * len(r_values))
    tao_seq = np.zeros((3, total_steps))
    current_idx = 0
    
    for i in range(len(u_values)):
        for j in range(len(r_values)):
            if current_idx >= total_steps:
                return tao_seq
            end_idx = min(current_idx + steps_per_combination, total_steps)
            tao_seq[0, current_idx:end_idx] = u_values[i]
            tao_seq[2, current_idx:end_idx] = r_values[j]
            current_idx = end_idx
    
    return tao_seq

def generate_combined_lateral(total_steps, dt, tau_v_lim, tau_r_lim):
    """横向+回转组合输入"""
    v_values = np.arange(tau_v_lim[0], tau_v_lim[1] + 10, 10)
    r_values = np.arange(tau_r_lim[0], tau_r_lim[1] + 10, 10)
    
    steps_per_combination = total_steps // (len(v_values) * len(r_values))
    tao_seq = np.zeros((3, total_steps))
    current_idx = 0
    
    for i in range(len(v_values)):
        for j in range(len(r_values)):
            if current_idx >= total_steps:
                return tao_seq
            end_idx = min(current_idx + steps_per_combination, total_steps)
            tao_seq[1, current_idx:end_idx] = v_values[i]
            tao_seq[2, current_idx:end_idx] = r_values[j]
            current_idx = end_idx
    
    return tao_seq

def generate_random_step(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim):
    """随机阶跃输入"""
    tao_seq = np.zeros((3, total_steps))
    min_interval = int(2 / dt)
    max_interval = int(3 / dt)
    
    current_idx = 0
    while current_idx < total_steps:
        interval = np.random.randint(min_interval, max_interval + 1)
        end_idx = min(current_idx + interval, total_steps)
        tao_seq[0, current_idx:end_idx] = np.random.uniform(tau_u_lim[0], tau_u_lim[1])
        tao_seq[1, current_idx:end_idx] = np.random.uniform(tau_v_lim[0], tau_v_lim[1])
        tao_seq[2, current_idx:end_idx] = np.random.uniform(tau_r_lim[0], tau_r_lim[1])
        current_idx = end_idx
    
    return tao_seq

def generate_sine_sweep(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim):
    """扫频信号"""
    tao_seq = np.zeros((3, total_steps))
    time = np.arange(total_steps) * dt
    
    freq_u = np.linspace(0.01, 0.1, total_steps)
    amp_u = (tau_u_lim[1] - tau_u_lim[0]) / 2
    tao_seq[0, :] = amp_u * np.sin(2 * np.pi * freq_u * time) + np.mean(tau_u_lim)
    
    freq_v = np.linspace(0.02, 0.15, total_steps)
    amp_v = (tau_v_lim[1] - tau_v_lim[0]) / 2
    tao_seq[1, :] = amp_v * np.sin(2 * np.pi * freq_v * time) + np.mean(tau_v_lim)
    
    freq_r = np.linspace(0.05, 0.2, total_steps)
    amp_r = (tau_r_lim[1] - tau_r_lim[0]) / 2
    tao_seq[2, :] = amp_r * np.sin(2 * np.pi * freq_r * time) + np.mean(tau_r_lim)
    
    return tao_seq

def generate_joint_excitation(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim):
    """联合激励"""
    tao_seq = np.zeros((3, total_steps))
    time = np.arange(total_steps) * dt
    
    for freq in [0.05, 0.1, 0.2, 0.3]:
        phase = 2 * np.pi * np.random.rand()
        tao_seq[0, :] += 0.3 * (tau_u_lim[1] - tau_u_lim[0]) * np.sin(2 * np.pi * freq * time + phase)
        tao_seq[1, :] += 0.3 * (tau_v_lim[1] - tau_v_lim[0]) * np.sin(2 * np.pi * freq * time + phase + np.pi/3)
        tao_seq[2, :] += 0.3 * (tau_r_lim[1] - tau_r_lim[0]) * np.sin(2 * np.pi * freq * time + phase + 2*np.pi/3)
    
    noise_level = 0.05
    tao_seq[0, :] += noise_level * np.random.randn(total_steps) * (tau_u_lim[1] - tau_u_lim[0])
    tao_seq[1, :] += noise_level * np.random.randn(total_steps) * (tau_v_lim[1] - tau_v_lim[0])
    tao_seq[2, :] += noise_level * np.random.randn(total_steps) * (tau_r_lim[1] - tau_r_lim[0])
    
    tao_seq[0, :] = np.clip(tao_seq[0, :], tau_u_lim[0], tau_u_lim[1])
    tao_seq[1, :] = np.clip(tao_seq[1, :], tau_v_lim[0], tau_v_lim[1])
    tao_seq[2, :] = np.clip(tao_seq[2, :], tau_r_lim[0], tau_r_lim[1])
    
    return tao_seq

def generate_orthogonal(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim):
    """正交激励"""
    steps_per_action = int(5 / dt)
    tao_seq = np.zeros((3, total_steps))
    
    num_actions = int(np.ceil(total_steps / steps_per_action))
    u_values = np.linspace(tau_u_lim[0], tau_u_lim[1], 10)
    v_values = np.linspace(tau_v_lim[0], tau_v_lim[1], 10)
    r_values = np.linspace(tau_r_lim[0], tau_r_lim[1], 10)
    
    for i in range(num_actions):
        start_idx = i * steps_per_action
        if start_idx >= total_steps:
            break
        end_idx = min((i + 1) * steps_per_action, total_steps)
        
        tao_seq[0, start_idx:end_idx] = u_values[(i-1) % len(u_values)]
        tao_seq[1, start_idx:end_idx] = v_values[i % len(v_values)]
        tao_seq[2, start_idx:end_idx] = r_values[(i+1) % len(r_values)]
    
    return tao_seq

# =============================== 网络定义 ===============================

class AUVLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(AUVLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.layernorm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.layernorm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.layernorm3 = nn.LayerNorm(64)
        
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.layernorm1(out)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.layernorm2(out)
        out = self.dropout2(out)
        
        out, _ = self.lstm3(out)
        out = self.layernorm3(out)
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

# =============================== 数据集定义 ===============================

class AUVDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])

# =============================== 训练函数 ===============================

def train_model(net, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, gradient_clip=1.0, patience=100):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs), desc='Training'):
        net.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                outputs = net(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = net.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    
    return net, train_losses, val_losses

# =============================== 可视化函数 ===============================

def plot_predictions(actual, predicted, dt):
    time = np.arange(len(actual)) * dt
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('LSTM模型预测性能对比 (测试集)', fontsize=16)
    
    axes[0, 0].plot(time, actual[:, 0], 'b', linewidth=1.5, label='实际值')
    axes[0, 0].plot(time, predicted[:, 0], 'r--', linewidth=1.5, label='预测值')
    axes[0, 0].set_title('纵向速度 u')
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('速度 (m/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time, actual[:, 1], 'b', linewidth=1.5, label='实际值')
    axes[0, 1].plot(time, predicted[:, 1], 'r--', linewidth=1.5, label='预测值')
    axes[0, 1].set_title('横向速度 v')
    axes[0, 1].set_xlabel('时间 (秒)')
    axes[0, 1].set_ylabel('速度 (m/s)')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(time, actual[:, 2], 'b', linewidth=1.5, label='实际值')
    axes[1, 0].plot(time, predicted[:, 2], 'r--', linewidth=1.5, label='预测值')
    axes[1, 0].set_title('回转角速度 r')
    axes[1, 0].set_xlabel('时间 (秒)')
    axes[1, 0].set_ylabel('角速度 (rad/s)')
    axes[1, 0].grid(True)
    
    psi_actual = wrap_to_2pi(actual[:, 3])
    psi_pred = wrap_to_2pi(predicted[:, 3])
    axes[1, 1].plot(time, psi_actual, 'b', linewidth=1.5, label='实际值')
    axes[1, 1].plot(time, psi_pred, 'r--', linewidth=1.5, label='预测值')
    axes[1, 1].set_title('航向角 ψ')
    axes[1, 1].set_xlabel('时间 (秒)')
    axes[1, 1].set_ylabel('角度 (rad)')
    axes[1, 1].grid(True)
    
    axes[2, 0].plot(time, actual[:, 4], 'b', linewidth=1.5, label='实际值')
    axes[2, 0].plot(time, predicted[:, 4], 'r--', linewidth=1.5, label='预测值')
    axes[2, 0].set_title('X位置')
    axes[2, 0].set_xlabel('时间 (秒)')
    axes[2, 0].set_ylabel('位置 (m)')
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(time, actual[:, 5], 'b', linewidth=1.5, label='实际值')
    axes[2, 1].plot(time, predicted[:, 5], 'r--', linewidth=1.5, label='预测值')
    axes[2, 1].set_title('Y位置')
    axes[2, 1].set_xlabel('时间 (秒)')
    axes[2, 1].set_ylabel('位置 (m)')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_all_training_data(data_types, all_tao_seqs, all_state_seqs, dt, num_simulations):
    for type_idx, data_type in enumerate(data_types):
        current_tao_seqs = all_tao_seqs[type_idx]
        current_state_seqs = all_state_seqs[type_idx]
        
        fig, axes = plt.subplots(3, num_simulations, figsize=(4*num_simulations, 10))
        fig.suptitle(f'{data_type} 训练数据 (所有仿真)', fontsize=16)
        
        for sim_idx in range(num_simulations):
            tao_seq = current_tao_seqs[sim_idx]
            state_seq = current_state_seqs[sim_idx]
            time = np.arange(tao_seq.shape[1]) * dt
            
            ax = axes[0, sim_idx] if num_simulations > 1 else axes[0]
            ax.plot(time, tao_seq[0, :], 'r', linewidth=1.5, label='纵向推力')
            ax.plot(time, tao_seq[1, :], 'g', linewidth=1.5, label='横向推力')
            ax.plot(time, tao_seq[2, :], 'b', linewidth=1.5, label='回转力矩')
            ax.set_title(f'控制输入 (仿真{sim_idx+1})')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('推力 (N)')
            ax.legend()
            ax.grid(True)
            
            ax = axes[1, sim_idx] if num_simulations > 1 else axes[1]
            ax.plot(time, state_seq[3, :], 'r', linewidth=1.5, label='纵向速度 u')
            ax.plot(time, state_seq[4, :], 'g', linewidth=1.5, label='横向速度 v')
            ax.plot(time, np.degrees(state_seq[5, :]), 'b', linewidth=1.5, label='回转速度 r')
            ax.set_title(f'速度 (仿真{sim_idx+1})')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('速度')
            ax.legend()
            ax.grid(True)
            
            ax = axes[2, sim_idx] if num_simulations > 1 else axes[2]
            ax.plot(time, state_seq[0, :], 'r', linewidth=1.5, label='X位置')
            ax.plot(time, state_seq[1, :], 'g', linewidth=1.5, label='Y位置')
            ax.plot(time, np.degrees(wrap_to_2pi(state_seq[2, :])), 'b', linewidth=1.5, label='航向角')
            ax.set_title(f'位置和航向 (仿真{sim_idx+1})')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_openloop_results(traj_true, traj_pred, dt, maneuver_name):
    time = np.arange(traj_true.shape[1]) * dt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{maneuver_name}开环验证 (直接位置预测)', fontsize=16)
    
    axes[0, 0].plot(traj_true[0, :], traj_true[1, :], 'b-', linewidth=2, label='物理模型')
    axes[0, 0].plot(traj_pred[0, :], traj_pred[1, :], 'r--', linewidth=2, label='LSTM模型')
    axes[0, 0].set_xlabel('X位置 (m)')
    axes[0, 0].set_ylabel('Y位置 (m)')
    axes[0, 0].set_title('位置轨迹')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True)
    
    psi_true = wrap_to_2pi(traj_true[2, :])
    psi_pred = wrap_to_2pi(traj_pred[2, :])
    axes[0, 1].plot(time, np.degrees(psi_true), 'b-', linewidth=1.5, label='物理模型')
    axes[0, 1].plot(time, np.degrees(psi_pred), 'r--', linewidth=1.5, label='LSTM模型')
    axes[0, 1].set_xlabel('时间 (秒)')
    axes[0, 1].set_ylabel('航向角 (度)')
    axes[0, 1].set_title('航向角变化')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(time, traj_true[3, :], 'b-', linewidth=1.5, label='物理模型')
    axes[1, 0].plot(time, traj_pred[3, :], 'r--', linewidth=1.5, label='LSTM模型')
    axes[1, 0].set_xlabel('时间 (秒)')
    axes[1, 0].set_ylabel('纵向速度 (m/s)')
    axes[1, 0].set_title('纵向速度')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, np.degrees(traj_true[5, :]), 'b-', linewidth=1.5, label='物理模型')
    axes[1, 1].plot(time, np.degrees(traj_pred[5, :]), 'r--', linewidth=1.5, label='LSTM模型')
    axes[1, 1].set_xlabel('时间 (秒)')
    axes[1, 1].set_ylabel('回转速度 (度/秒)')
    axes[1, 1].set_title('回转速度')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{maneuver_name}开环测试误差分析', fontsize=16)
    
    pos_error_x = traj_true[0, :] - traj_pred[0, :]
    pos_error_y = traj_true[1, :] - traj_pred[1, :]
    axes[0, 0].plot(time, pos_error_x, 'r', linewidth=1.5, label='X位置误差')
    axes[0, 0].plot(time, pos_error_y, 'b', linewidth=1.5, label='Y位置误差')
    axes[0, 0].set_title('位置误差')
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('误差 (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    vel_error_u = traj_true[3, :] - traj_pred[3, :]
    vel_error_v = traj_true[4, :] - traj_pred[4, :]
    vel_error_r = np.degrees(traj_true[5, :] - traj_pred[5, :])
    axes[0, 1].plot(time, vel_error_u, 'r', linewidth=1.5, label='纵向速度 u')
    axes[0, 1].plot(time, vel_error_v, 'g', linewidth=1.5, label='横向速度 v')
    axes[0, 1].plot(time, vel_error_r, 'b', linewidth=1.5, label='回转速度 r')
    axes[0, 1].set_title('速度误差')
    axes[0, 1].set_xlabel('时间 (秒)')
    axes[0, 1].set_ylabel('误差')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    psi_true_wrapped = wrap_to_2pi(traj_true[2, :])
    psi_pred_wrapped = wrap_to_2pi(traj_pred[2, :])
    psi_diff = np.arctan2(np.sin(psi_true_wrapped - psi_pred_wrapped), 
                          np.cos(psi_true_wrapped - psi_pred_wrapped))
    psi_error = np.degrees(psi_diff)
    axes[1, 0].plot(time, psi_error, 'm', linewidth=1.5)
    axes[1, 0].set_title('航向角误差')
    axes[1, 0].set_xlabel('时间 (秒)')
    axes[1, 0].set_ylabel('误差 (度)')
    axes[1, 0].grid(True)
    
    cumulative_error = np.sqrt(np.sum((traj_true[0:2, :] - traj_pred[0:2, :])**2, axis=0))
    axes[1, 1].plot(time, cumulative_error, 'k', linewidth=2)
    axes[1, 1].set_title('累积位置误差')
    axes[1, 1].set_xlabel('时间 (秒)')
    axes[1, 1].set_ylabel('误差 (m)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    pos_error = traj_true[0:2, :] - traj_pred[0:2, :]
    rmse_pos = np.sqrt(np.mean(pos_error**2, axis=1))
    
    vel_error = traj_true[3:6, :] - traj_pred[3:6, :]
    rmse_vel = np.sqrt(np.mean(vel_error**2, axis=1))
    
    psi_rmse = np.sqrt(np.mean(psi_diff**2))
    
    print(f'\n{maneuver_name}性能指标:')
    print(f'  位置RMSE: X={rmse_pos[0]:.4f} m, Y={rmse_pos[1]:.4f} m')
    print(f'  速度RMSE: u={rmse_vel[0]:.4f} m/s, v={rmse_vel[1]:.4f} m/s, r={rmse_vel[2]:.4f} rad/s')
    print(f'  航向角RMSE: {psi_rmse:.4f} rad ({np.degrees(psi_rmse):.2f} deg)')

def run_open_loop_simulation(net, tao, duration, dt, seq_length, 
                             x_mean, x_std, y_mean, y_std, 
                             u_lim, v_lim, r_lim, pos_lim, device):
    """开环仿真核心函数"""
    steps = int(duration / dt)
    traj_true = np.zeros((6, steps))
    traj_pred = np.zeros((6, steps))
    
    xx_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xx_pred = xx_true.copy()
    
    input_buffer = np.zeros((9, seq_length))
    
    for i in range(seq_length):
        input_buffer[:, i] = [
            tao[0], tao[1], tao[2],
            xx_pred[3], xx_pred[4], xx_pred[5],
            wrap_to_2pi_scalar(xx_pred[2]),
            xx_pred[0], xx_pred[1]
        ]
    
    for k in range(steps):
        xx_true = auv_3dof_euler_2input_limited(xx_true, tao, dt, u_lim, v_lim, r_lim)
        traj_true[:, k] = xx_true
        
        input_seq = input_buffer.T
        input_seq_norm = (input_seq - x_mean) / x_std
        
        input_tensor = torch.FloatTensor(input_seq_norm).unsqueeze(0).to(device)
        
        with torch.no_grad():
            Y_pred_norm = net(input_tensor)
            Y_pred = (Y_pred_norm.cpu().numpy().squeeze() * y_std) + y_mean
        
        u_next, v_next, r_next, psi_next, x_next, y_next = Y_pred
        psi_next = wrap_to_2pi_scalar(psi_next)
        
        xx_pred[0] = x_next
        xx_pred[1] = y_next
        xx_pred[2] = psi_next
        xx_pred[3] = u_next
        xx_pred[4] = v_next
        xx_pred[5] = r_next
        
        xx_pred[0] = np.clip(xx_pred[0], pos_lim[0], pos_lim[1])
        xx_pred[1] = np.clip(xx_pred[1], pos_lim[0], pos_lim[1])
        
        traj_pred[:, k] = xx_pred
        
        input_buffer[:, :-1] = input_buffer[:, 1:]
        input_buffer[:, -1] = [
            tao[0], tao[1], tao[2],
            xx_pred[3], xx_pred[4], xx_pred[5],
            wrap_to_2pi_scalar(xx_pred[2]),
            xx_pred[0], xx_pred[1]
        ]
        
        if k == 0:
            print(f'第一步预测值: [u={u_next:.4f}, v={v_next:.4f}, r={r_next:.4f} rad/s]')
            print(f'第一步真实值: [u={xx_true[3]:.4f}, v={xx_true[4]:.4f}, r={xx_true[5]:.4f} rad/s]')
    
    return traj_true, traj_pred

def validate_straight_turn(net, x_mean, x_std, y_mean, y_std, dt, seq_length, 
                           u_lim, v_lim, r_lim, pos_lim, device):
    print('验证直航工况...')
    traj_true, traj_pred = run_open_loop_simulation(
        net, np.array([120.0, 0.0, 0.0]), 120, dt, seq_length,
        x_mean, x_std, y_mean, y_std, u_lim, v_lim, r_lim, pos_lim, device
    )
    plot_openloop_results(traj_true, traj_pred, dt, '直航工况')
    
    print('验证回转工况...')
    traj_true, traj_pred = run_open_loop_simulation(
        net, np.array([100.0, 0.0, 10.0]), 120, dt, seq_length,
        x_mean, x_std, y_mean, y_std, u_lim, v_lim, r_lim, pos_lim, device
    )
    plot_openloop_results(traj_true, traj_pred, dt, '回转工况')

# =============================== 主函数 ===============================

def main():
    # 参数设置
    dt = 0.1
    sim_time = 120
    total_steps = int(sim_time / dt)
    seq_length = 20
    
    tau_u_lim = [-200, 200]
    tau_v_lim = [-150, 150]
    tau_r_lim = [-50, 50]
    
    u_lim = [-4.6, 4.6]
    v_lim = [-0.5, 0.5]
    r_lim = np.radians([-60, 60])
    pos_lim = [-1000, 1000]
    
    data_types = [
        'constant_single',
        'combined_rotation',
        'combined_lateral',
        'random_step',
        'sine_sweep',
        'joint_excitation',
        'orthogonal'
    ]
    
    num_simulations = 5
    
    # 数据生成
    print("开始数据生成...")
    all_inputs = []
    all_states = []
    all_next_states = []
    all_tao_seqs = [[] for _ in range(len(data_types))]
    all_state_seqs = [[] for _ in range(len(data_types))]
    
    for type_idx, data_type in enumerate(data_types):
        print(f'生成数据类型: {data_type} ({num_simulations}次仿真)')
        
        current_tao_seqs = []
        current_state_seqs = []
        
        for sim_idx in range(num_simulations):
            xx = np.array([
                np.random.randint(-100, 101),
                np.random.randint(-100, 101),
                0.0, 0.0, 0.0, 0.0
            ], dtype=float)
            
            if data_type == 'constant_single':
                tao_seq = generate_constant_single(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim)
            elif data_type == 'combined_rotation':
                tao_seq = generate_combined_rotation(total_steps, dt, tau_u_lim, tau_r_lim)
            elif data_type == 'combined_lateral':
                tao_seq = generate_combined_lateral(total_steps, dt, tau_v_lim, tau_r_lim)
            elif data_type == 'random_step':
                tao_seq = generate_random_step(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim)
            elif data_type == 'sine_sweep':
                tao_seq = generate_sine_sweep(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim)
            elif data_type == 'joint_excitation':
                tao_seq = generate_joint_excitation(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim)
            elif data_type == 'orthogonal':
                tao_seq = generate_orthogonal(total_steps, dt, tau_u_lim, tau_v_lim, tau_r_lim)
            
            current_tao_seqs.append(tao_seq)
            
            state_seq = np.zeros((6, total_steps))
            input_seq = np.zeros((3, total_steps))
            next_state_seq = np.zeros((6, total_steps))
            
            for step in range(total_steps):
                tao = tao_seq[:, step]
                next_xx = auv_3dof_euler_2input_limited(xx, tao, dt, u_lim, v_lim, r_lim)
                
                noise_level = 0.001
                state_noise = noise_level * np.random.randn(6) * np.array([0.1, 0.1, 0.01, 0.1, 0.02, 0.05])
                next_xx += state_noise
                
                state_seq[:, step] = xx
                input_seq[:, step] = tao
                next_state_seq[:, step] = next_xx
                
                if step == 0 and sim_idx == 0 and type_idx == 0:
                    print(f'初始状态: [u={xx[3]:.2f}, v={xx[4]:.2f}, r={xx[5]:.2f} rad/s]')
                    print(f'控制输入: [tau_u={tao[0]:.2f}, tau_v={tao[1]:.2f}, tau_r={tao[2]:.2f}]')
                    print(f'下一状态: [u={next_xx[3]:.2f}, v={next_xx[4]:.2f}, r={next_xx[5]:.2f} rad/s]')
                
                xx = next_xx.copy()
                xx[2] = wrap_to_2pi_scalar(xx[2])
            
            # 航向角连续性处理
            psi_full = np.concatenate([state_seq[2, :], [next_state_seq[2, -1]]])
            psi_full_unwrap = np.unwrap(psi_full)
            state_seq[2, :] = psi_full_unwrap[:-1]
            next_state_seq[2, :] = psi_full_unwrap[1:]
            
            state_seq[2, :] = wrap_to_2pi(state_seq[2, :])
            next_state_seq[2, :] = wrap_to_2pi(next_state_seq[2, :])
            
            all_inputs.append(input_seq)
            all_states.append(state_seq)
            all_next_states.append(next_state_seq)
            
            current_state_seqs.append(state_seq)
            
            print(f'  完成仿真 {sim_idx+1}/{num_simulations}')
        
        all_tao_seqs[type_idx] = current_tao_seqs
        all_state_seqs[type_idx] = current_state_seqs
    
    all_inputs = np.hstack(all_inputs)
    all_states = np.hstack(all_states)
    all_next_states = np.hstack(all_next_states)
    
    # 训练数据可视化
    print('可视化所有训练数据...')
    #visualize_all_training_data(data_types, all_tao_seqs, all_state_seqs, dt, num_simulations)
    
    # 序列数据生成
    print(f'生成序列数据 (序列长度={seq_length})...')
    total_samples = all_inputs.shape[1] - seq_length
    print(f'总样本数: {total_samples}')
    
    X_sequence = np.zeros((9, seq_length, total_samples))
    Y_target = np.zeros((6, total_samples))
    
    progress_step = max(1, total_samples // 100)
    print('进度: 0%', end='')
    
    for i in range(total_samples):
        start_idx = i
        end_idx = i + seq_length
        next_idx = i + seq_length
        
        input_seq = np.vstack([
            all_inputs[:, start_idx:end_idx],
            all_states[3:6, start_idx:end_idx],
            all_states[2, start_idx:end_idx].reshape(1, -1),
            all_states[0:2, start_idx:end_idx]
        ])
        
        target = np.concatenate([
            all_next_states[3:6, next_idx],
            [all_next_states[2, next_idx]],
            all_next_states[0:2, next_idx]
        ])
        
        X_sequence[:, :, i] = input_seq
        Y_target[:, i] = target
        
        if (i + 1) % progress_step == 0:
            print(f'\r进度: {int((i+1)/total_samples*100)}%', end='')
    
    print('\r进度: 100%')
    X_sequence = np.transpose(X_sequence, (2, 1, 0))
    Y_target = Y_target.T
    
    print(f'序列数据集大小: 输入={X_sequence.shape}, 输出={Y_target.shape}')
    
    # 数据标准化
    print('数据标准化处理...')
    num_samples, seq_len, num_features = X_sequence.shape
    
    X_reshaped = X_sequence.reshape(-1, num_features)
    Y_reshaped = Y_target
    
    x_mean = np.mean(X_reshaped, axis=0)
    x_std = np.std(X_reshaped, axis=0)
    y_mean = np.mean(Y_reshaped, axis=0)
    y_std = np.std(Y_reshaped, axis=0)
    
    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1
    
    X_normalized = (X_reshaped - x_mean) / x_std
    Y_normalized = (Y_reshaped - y_mean) / y_std
    
    X_normalized = X_normalized.reshape(num_samples, seq_len, num_features)
    
    # 数据集分割
    print('分割数据集...')
    shuffle_idx = np.random.permutation(num_samples)
    X_normalized = X_normalized[shuffle_idx]
    Y_normalized = Y_normalized[shuffle_idx]
    
    train_end = int(num_samples * 0.8)
    val_end = train_end + int(num_samples * 0.1)
    
    X_train, Y_train = X_normalized[:train_end], Y_normalized[:train_end]
    X_val, Y_val = X_normalized[train_end:val_end], Y_normalized[train_end:val_end]
    X_test, Y_test = X_normalized[val_end:], Y_normalized[val_end:]
    
    print(f'训练集: {X_train.shape[0]} 样本')
    print(f'验证集: {X_val.shape[0]} 样本')
    print(f'测试集: {X_test.shape[0]} 样本')
    
    # 创建DataLoader
    batch_size = 128
    
    train_dataset = AUVDataset(X_train, Y_train)
    val_dataset = AUVDataset(X_val, Y_val)
    test_dataset = AUVDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 网络训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    input_size = X_train.shape[2]
    output_size = Y_train.shape[1]
    
    net = AUVLSTM(input_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=50)
    
    print('开始训练LSTM网络...')
    net, _, _ = train_model(
        net, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=15, device=device, gradient_clip=1.0, patience=100
    )
    
    # 模型评估
    print('在测试集上评估模型...')
    net.eval()
    Y_pred_list = []
    Y_test_list = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_pred = net(X_batch)
            Y_pred_list.append(Y_pred.cpu().numpy())
            Y_test_list.append(Y_batch.numpy())
    
    Y_pred_norm = np.vstack(Y_pred_list)
    Y_test_norm = np.vstack(Y_test_list)
    
    Y_pred = Y_pred_norm * y_std + y_mean
    Y_test_actual = Y_test_norm * y_std + y_mean
    
    rmse = np.sqrt(np.mean((Y_test_actual - Y_pred)**2, axis=0))
    print('测试集RMSE:')
    for i, name in enumerate(['u', 'v', 'r', 'psi', 'x', 'y']):
        print(f'  {name}: {rmse[i]:.4f}')
    
    ranges = np.max(Y_test_actual, axis=0) - np.min(Y_test_actual, axis=0)
    relative_rmse = rmse / ranges
    print('相对RMSE:')
    for i, name in enumerate(['u', 'v', 'r', 'psi', 'x', 'y']):
        print(f'  {name}: {relative_rmse[i]*100:.2f}%')
    
    plot_predictions(Y_test_actual, Y_pred, dt)
    
    # 保存模型
    model_save_path = 'auv_lstm_enhanced_direct_position_no_wrap.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'seq_length': seq_length
    }, model_save_path)
    print(f'模型已保存为 {model_save_path}')
    
    # 开环验证
    print('开始开环验证...')
    validate_straight_turn(
        net, x_mean, x_std, y_mean, y_std, dt, seq_length,
        u_lim, v_lim, r_lim, pos_lim, device
    )

if __name__ == '__main__':
    main()