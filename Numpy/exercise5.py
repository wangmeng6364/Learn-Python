# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

# 计算逆矩阵
'''A = np.mat("0 1 2; 1 0 3; 4 -3 8")
print "A", A
inverse = np.linalg.inv(A)
print "Inverse of A", inverse
print "Check", A * inverse'''

# 求解线性方程组(Bx=b)
'''B = np.mat("1 -2 1; 0 2 -8; -4 5 9")
print "B", B
b = np.array([0, 8, 9])
print "b", b
x = np.linalg.solve(B, b)
print "Solution", x
print "Check", np.dot(B, x)'''

# 特征值和特征向量(Ax=bx)
'''C = np.mat('3 -2; 1 0')
print "C", C
# eigvals()求解特征值
print "Eigenvalues", np.linalg.eigvals(C)
# eig()求解特征值和特征向量，返回元祖
eigenvalues, eigenvectors = np.linalg.eig(C)
print "Eigenvalues", eigenvalues
print "Eigenvectors", eigenvectors
for i in range(len(eigenvalues)):
    print "Left", np.dot(C, eigenvectors[:,i])
    print "Right", eigenvalues[i] * eigenvectors'''
    
# 奇异值分解
'''D = np.mat('4 11 14; 8 7 -2')
print "D", D
U, Sigma, V = np.linalg.svd(D, full_matrices=False)
print "U", U
print "Sihma", Sigma  # 得到的Sigma只是奇异矩阵的对角值
print "V", V
# 通过diag生成真正的SVD
true_svd = np.diag(Sigma)
print "SVD", true_svd
print "Check", U * true_svd * V'''

# 广义逆矩阵
'''E = np.mat('4 11 14; 8 7 -2')
print "E", E
# 计算广义逆矩阵使用pinv()
pseudoinv = np.linalg.pinv(E)
print "Pseudo inverse", pseudoinv
print "Check", E * pseudoinv'''

# 行列式
'''F = np.mat('3 4; 5 6')
print "F", F
print "Determinant", np.linalg.det(F)'''

# 快速傅里叶变换
'''x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
i_transformed = np.fft.ifft(transformed)
plt.plot(x, wave)
plt.plot(x, transformed)
plt.plot(x, i_transformed)
plt.show()'''

# 移频
'''x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
shifted = np.fft.fftshift(transformed)
i_shifted = np.fft.ifftshift(shifted)
plt.plot(x, transformed)
plt.plot(x, shifted)
plt.plot(x, i_shifted)
plt.show()'''

# 二项分布
'''cash = np.zeros(10000)
cash[0] = 1000
# binomial()函数
outcome = np.random.binomial(9, 0.5, size=len(cash))
for i in range(1, len(cash)):
    if 0 <= outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError("Unexpected outcome " + outcome)
print outcome.min(), outcome.max()
plt.plot(np.arange(len(cash)), cash)
plt.show()'''

# 超几何分布
'''points = np.zeros(100)
outcome = np.random.hypergeometric(25, 1, 3, size=len(points))
for i in np.arange(1, len(points)):
    if outcome[i] == 3: 
        points[i] = points[i-1] + 1
    elif outcome[i] == 2: 
        points[i] = points[i-1] - 6
print outcome.min(), outcome.max()
plt.plot(np.arange(len(points)), points)
plt.show()'''

# 正态分布
'''# 首先产生一定数量的随机数
N = 10000
normal_values = np.random.normal(size=N)
# 第二个参数bins默认为10，即柱子的数量，normed=True计算密度，而非次数
dummy, bins, dummy = plt.hist(normal_values, np.sqrt(N), normed=True, lw=1.0) 
sigma = 1
mu = 0
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins-mu)**2/(2 * sigma ** 2)), lw=2.0)
plt.show()'''


