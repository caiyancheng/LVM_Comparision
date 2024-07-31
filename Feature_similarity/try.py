import numpy as np
import matplotlib.pyplot as plt

# 示例二维矩阵
data = np.array([[1, 2, 3, 4],
                 [2, 1.9, 4, 5],
                 [3, 4, 5, 6],
                 [4, 5, 6.2, 7]])

# 创建网格
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# 绘制等高线图
plt.contour(X, Y, data)
plt.colorbar(label='Value')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Contour Plot')
plt.show()
