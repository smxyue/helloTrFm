import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False    # Fix negative sign issue

# 创建示例数据
# 灰度图像数据 (2D数组)
gray_data = np.random.rand(100, 100) * 255
gray_data = gray_data.astype(np.uint8)

# RGB图像数据 (3D数组: 高×宽×通道)
rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
rgb_data[30:70, 30:70, 0] = 255  # 红色通道
rgb_data[20:80, 40:60, 1] = 255  # 绿色通道
rgb_data[40:60, 20:80, 2] = 255  # 蓝色通道

# 折线图数据
x_line = np.linspace(0, 10, 100)
y_line = np.sin(x_line) * np.exp(-x_line * 0.1)

# 条形图数据
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 100, size=5)

# 创建主图形
fig = plt.figure(figsize=(12, 10))

# 1. 显示灰度图像
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(gray_data, cmap='gray')
ax1.set_title('灰度图像', fontsize=14)
ax1.set_xlabel('宽度 (像素)')
ax1.set_ylabel('高度 (像素)')

# 2. 显示折线图
ax2 = plt.subplot(2, 2, 2)
ax2.plot(x_line, y_line, color='blue', linewidth=2, marker='.', markersize=4)
ax2.set_title('衰减正弦波', fontsize=14)
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('振幅')
ax2.grid(True, alpha=0.3)

# 3. 显示条形图
ax3 = plt.subplot(2, 2, 3)
bars = ax3.bar(categories, values, color='skyblue', edgecolor='navy')
ax3.set_title('类别统计', fontsize=14)
ax3.set_xlabel('类别')
ax3.set_ylabel('数值')
# 在柱子上方显示数值
for bar, value in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(value), ha='center', va='bottom')

# 4. 显示RGB彩色图像
ax4 = plt.subplot(2, 2, 4)
ax4.imshow(rgb_data)
ax4.set_title('RGB彩色图像', fontsize=14)
ax4.set_xlabel('宽度 (像素)')
ax4.set_ylabel('高度 (像素)')

# 调整布局并显示
plt.tight_layout()
plt.show()

# 独立保存各图的方法
# 灰度图单独保存
plt.figure()
plt.imshow(gray_data, cmap='gray', vmin=0, vmax=255)
plt.colorbar(label='灰度值')
plt.title('灰度图示例')
plt.savefig('gray_image.png', dpi=150, bbox_inches='tight')
plt.close()

# 折线图单独保存
plt.figure()
plt.plot(x_line, y_line, 'r-o')
plt.title('折线图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid()
plt.savefig('line_plot.png', dpi=150, bbox_inches='tight')
plt.close()