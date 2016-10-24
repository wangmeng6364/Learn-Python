# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

# 创建矩阵
'''A = np.mat('1 2 3; 4 5 6; 7 8 9')
print "Creation from string", A
# T属性即为转置矩阵
print "Transpose", A.T
# I属性为逆矩阵
print "Inverse", A.I
# 通过Numpy数组创建
print "Creation from array", np.mat(np.arange(9).reshape(3, 3))'''

# 从已有矩阵创建新矩阵
'''A = np.eye(2)
print 'A', A
B = A * 2
print "B", B
print "Compound matrix\n", np.bmat("A B; A B")  # 有点像分块矩阵'''

# 除法运算
'''a = np.array([2, 6, 5])
b = np.array([1, 2, 3])
print "Divide", np.divide(a, b), np.divide(b, a)  # 相当于"/"
print "True Divide", np.true_divide(a, b), np.true_divide(b, a)  
print "Floor Divide", np.floor_divide(a, b), np.floor_divide(b, a)  # 相当于"//"'''

# 模运算
'''a = np.arange(-4, 4)
print "a", a
print "Remainder", np.remainder(a, 2)  # 相当于"%", mod
print "Fmod", np.fmod(a, 2)  # fmod的区别在于处理负数的方式'''

# 创建斐波那契数列
'''F = np.matrix([[1, 1], [1, 0]])   # 特殊的矩阵
print "F", F
print "8th Fibonacci", (F**7)[0, 0]'''

# 绘制利萨如曲线
# 初始化相关参数
a = 9
b = 8
n = np.pi
t = np.linspace(-np.pi, np.pi, 201)  # 产生-pi~pi均匀分布的201个点
x = np.sin(a * t + n / 2)
y = np.sin(b * t)
plt.plot(x, y)
plt.show()
