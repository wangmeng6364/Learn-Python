# -*- coding:utf-8 -*-
# 先导入numpy库
import numpy as np
# arange()的用法与range()类似
a = np.arange(1, 10, 2)
print 'a:', a
# 利用array()可以直接创建
b = np.array([[1, 2], [1, 2]])
print 'b:', b
# arange()与array()结合
c = np.array([np.arange(2), np.arange(2)])
print 'c:', c

d = np.array([[1, 2], [3, 4]]) # 这是一个2*2的数组
print d
# 选取第一行第一个元素
print d[0, 0]
# 选取第二行第一个元素
print d[1, 0]

e = np.arange(10)
print e
print e[1:5:2]

# reshape()可以改变数组的维度
f = np.arange(24).reshape(2, 3, 4)
print u'数组', f

# 可以通过shape属性查看数据的维度信息
#print f.shape

# 把数组f理解为一个二层的楼，每一层有3行4列个房间
print u'选取第一层的所有房间', f[0] # 等价于f[0, :, :]和f[0, ...]

print u'选取第一层楼第二行的房间', f[0, 1]  # 等价于f[0, 1, :]

# 也可以按照一定的步长
print u'选取第一层楼第一和第三行', f[0, 0:3:2]

# 展平操作，两种方式
g = np.array([np.arange(5), np.arange(5)])
print u"数组g:", g
print "ravel:", g.ravel()
print "flatten:", g.flatten()

# 设置维度，三种方式
print "reshape:", g.reshape(5, 2)
print u"数组g:", g       # g本身并没有变
# 以下两种方式数组g被改变
g = np.array([np.arange(5), np.arange(5)])
g.shape = (5, 2)
print "shape:", g
g = np.array([np.arange(5), np.arange(5)]) 
g.resize((5, 2))
print "resize:", g 

# 转置
# 数组本身有一个T属性
g = np.array([np.arange(5), np.arange(5)])
print "T:", g.T
# 可以用transpose方法
g = np.array([np.arange(5), np.arange(5)])
print "transpose:", g.transpose()

# 数组的组合
# 先创建两个数组
h = np.arange(9).reshape(3, 3)
print u"数组h:", h
i = 2 * h
print u"数组i:", i

# 水平组合hstack
print "hstack:", np.hstack((h, i))
# concatenate函数也可以，不过要设置参数
print "concatenate(axis=1):", np.concatenate((h, i), axis=1)

# 垂直组合vstack
print "vstack:", np.vstack((h, i))
# 当然也可以用concatenate，axis默认为0
print "concatenate(axis=0):", np.concatenate((h, i))

# 深度组合dstack，像是切蛋糕
print "dstack:", np.dstack((h, i))

# 列组合column_stack, 二维数组column_stack与hstack一样
print "column_stack:", np.column_stack((h, i))
# 行组合row_stack, 二维数组row_stack与vstack一样
print "row_stack:", np.row_stack((h, i))

# 数组的分割
j = np.arange(9).reshape(3, 3)
print u"数组j:", j

# 水平分割hsplit，可以理解为沿着水平方向进行分割
print "hsplit:", np.hsplit(j, 3)
# 也可以用split，指定axis=1
print "split(axis=1):", np.split(j, 3, axis=1)

# 垂直分割vsplit，可以理解为沿着垂直方向进行分割
print "vsplit:", np.vsplit(j, 3)
# 同样的split也可以，axis默认即为0
print "split(axis=0):", np.split(j, 3)

# 深度分割dsplit，至少三维
k = np.arange(27).reshape(3, 3, 3)
print k
print "dsplit:", np.dsplit(k, 3)

l = np.arange(20).reshape(2, 10)
print u"数组l:", l

print u"数组的维数:", l.ndim

print u"数组的维度:", l.shape

print u"数组元素的个数:", l.size

print u"数组单个元素所占内存:", l.itemsize

print u"数组所占内存:", l.nbytes

print u"转置:", l.T

print u"数组元素类型:", l.dtype

# flat属性返回一个flatier对象(扁平迭代器)
f = l.flat
print f
# 遍历
for item in f:
    print item
# 直接获取
print "l.flat[2]:", l.flat[2]
print "l.flat[[1, 2, 3]]:", l.flat[[1, 2, 3]]
# 可赋值
l.flat[[1, 2, 3]] = 1 
print u"flat赋值:", l

# 数组的转换
# numpy数组转换为list
m = np.arange(5)
print m.tolist()
# astype可以指定数据类型
print m.astype(complex)











