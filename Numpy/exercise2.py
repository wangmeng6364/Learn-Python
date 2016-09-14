# -*- coding:utf-8 -*-
# 导入numpy库
import numpy as np
import matplotlib.pyplot as plt

'''# 文件读写
# 先创建一个单位矩阵
a = np.eye(2)   # numpy自带的eye方法
print a
# 使用savetxt函数保存
np.savetxt("D:\Learn\Code\python\exercise\eye.txt", a)'''

'''# CSV文件
# delimiter定义分隔符，默认是空格；usecols定义选取的；unpack默认False，解包
c, v = np.loadtxt('D:\Learn\Code\python\exercise\data.csv', \
                   delimiter=",", usecols=(6, 7), unpack=True)
print u"收盘价:", c
print u"成交量:", v'''

'''# 成交量平均价格
c, v = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(6, 7), unpack=True)
# 计算以成交量加权的价格
vwap = np.average(c, weights=v)
print "VWAP:", vwap

# 算术平均价格
mean = np.mean(c)
print "Mean:", mean

# 时间加权平均价格
# 先构造一个时间序列
t = np.arange(len(c))
twap = np.average(c, weights=t)
print "TWAP:", twap'''

'''# 载入每日最高价和最低价的数据
h, l = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(4, 5), unpack=True)
# 直接调用max()和min()函数
print "highest:", np.max(h)
print "lowest:", np.min(l)

# ptp函数可以计算取值范围
print "Spred high price:", np.ptp(h)    
print "Spred low price:", np.ptp(l)     '''      

'''# 简单统计分析
c = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(6, ), unpack=True)
# 寻找中位数
print "median:", np.median(c)
# 利用排序来检验正确与否
sorted_c = np.msort(c)
print "sorted_c:", sorted_c
N = len(c) 
if N%2 != 0:
    middle = sorted_c[N/2]
else:
    middle = (sorted_c[N/2] + sorted_c[N/2-1])/2
print "middle:", middle

# 计算方差
variance = np.var(c)
print "variance:", variance'''

'''# 股票收益率
c = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(6, ), unpack=True)
# 简单收益率
# diff函数返回相邻数组元素差值组成的数组
diff = np.diff(c)
print "diff:", diff   
# 收益率等于这一天与前一天的差值除以前一天的值
returns = diff/c[:-1]
print "returns:", returns
# 计算收益率的标注差
print "Standard deviation:", np.std(returns)

# 对数收益率
logreturns = np.diff(np.log(c))
print "logreturns:", logreturns
print "Standard deviation:", np.std(logreturns) 

# 计算收益率为正值的情况
posretindices = np.where(returns > 0)
print "Indices with positive returns:", posretindices

# 波动率（对数收益率的标准差除以其均值，再除以交易日倒数的平方根，交易日通常取252天）
annual_volatility = np.std(logreturns)/np.mean(logreturns)/np.sqrt(1.0/252)
print "Annual volatility:", annual_volatility
print "Month volatility:", annual_volatility * np.sqrt(1.0/12)'''

'''# 时间处理
# 先导入时间模块
import datetime
# 编写转换函数
def datetostr(s):
    truetime = datetime.datetime.strptime(s, '%d-%m-%Y').weekday()
    return truetime
# 开始读取数据，并添加converters参数    
dates, close = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(1, 6), converters={1: datetostr}, \
                   unpack=True)
print "Dates:", dates
# 创建一个包含五个元素的数组
averages = np.zeros(5)
# where会根据条件返回满足条件的元素索引，take可以从索引中取出数据
for i in range(5):
    indices = np.where(dates == i)
    print indices
    prices = np.take(close, indices)
    avg = np.mean(prices)
    print "Day", i, "prices", prices, "Average", avg
    averages[i] = avg'''
    
# 周汇总
'''import datetime
def datetostr(s):
    truetime = datetime.datetime.strptime(s, '%d-%m-%Y').weekday()
    return truetime
dates, open, high, low, close = np.loadtxt("D:\Learn\Code\python\exercise\data.csv",\
                   delimiter=",", usecols=(1, 3, 4, 5, 6), converters={1: datetostr}, \
                   unpack=True)   
# 为了方便计算，仅取前三周数据
dates = dates[:16]  
               
# 寻找第一个星期一
first_monday = np.ravel(np.where(dates == 0))[0] # where返回的是个多维数组，需要展平
print "First Monday", first_monday
# 寻找最后一个星期五            
last_friday = np.ravel(np.where(dates == 4))[-1]
print "Last Friday"

weeks_indices = np.arange(first_monday, last_friday+1)
print "Weeks indices initial", weeks_indices

weeks_indices = np.split(weeks_indices, 3)
print "Weeks indices after split", weeks_indices

# 为了后面的apply_along_axis
def summarize(a, o, h, l, c):
    monday_open = o[a[0]]
    week_high = np.max(np.take(h, a))
    week_low = np.min(np.take(l, a))
    friday_close = c[a[-1]]
    return ("APPL", monday_open, week_high, week_low, friday_close)
# apply_along_axis内涵很丰富  
weeksummary = np.apply_along_axis(summarize, 1, weeks_indices, open, high, low, close)
print "Week summary", weeksummary
# savetxt参数其实有很多
np.savetxt("D:\Learn\Code\python\exercise\weekssummary.csv", weeksummary, delimiter=",", fmt="%s")'''

# 真实波动幅度均值
'''h, l, c = np.loadtxt('D:\Learn\Code\python\exercise\data.csv', delimiter=',', usecols=(4, 5, 6), unpack=True)

N = 20
# 切片
h = h[-N:]
l = l[-N:]

print "len(h)", len(h), "len(l)", len(l)
print "Close", c
# 计算前一日的收盘价
previousclose = c[-N-1:-1]

print "len(previousclose)", len(previousclose)
print "Previous close", previousclose
# maximum函数可以选择出每个元素的最大值
truerange = np.maximum(h-l, h-previousclose, previousclose-l)

print "True range", truerange
# zeros函数初始化数组为0
atr = np.zeros(N)

atr[0] = np.mean(truerange)

for i in range(1, N):
    atr[i] = (N-1)*atr[i-1] + truerange[i]
    atr[i] /= N

print "ATR", atr   '''

# 简单移动平均线
'''# N是移动窗口的大小
N = 5 
# 权重是个平均值
weights = np.ones(N) / N
print "Weights", weights

c = np.loadtxt('D:\Learn\Code\python\exercise\data.csv', delimiter=',', usecols=(6, ), unpack=True)
# 要从卷积运算中取出与原数组重叠的区域
sma = np.convolve(weights, c)[N-1:-N+1]
# 生成一个时间序列
t = np.arange(N-1, len(c))
# 用matplotlib绘图
plt.plot(t, c[N-1:],'b-', lw=1.0)
plt.plot(t, sma, 'r-', lw=2.0)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title(u"Simple Moving Average")
plt.annotate('before convolve', xy=(12.8, 363), xytext=(15, 363),
arrowprops=dict(facecolor='black',shrink=0.005))
plt.annotate('after convolve', xy=(15, 358), xytext=(17, 358),
arrowprops=dict(facecolor='black',shrink=0.005))
plt.show()    '''

# 指数移动平均线
'''x = np.arange(5)
# 对x求指数，exp函数
print "Exp", np.exp(x)
# linspace函数实现等距分隔
print "Linspace", np.linspace(-1, 0 ,5)
N = 5
# 上面是两个示范，下面才是真的
weights = np.exp(np.linspace(-1, 0, N))
weights /= weights.sum()
print "Weights", weights

c = np.loadtxt('D:\Learn\Code\python\exercise\data.csv', delimiter=',', usecols=(6, ), unpack=True)
ema = np.convolve(weights, c)[N-1:-N+1]
t = np.arange(N-1, len(c))
plt.plot(t, c[N-1:],'b-', lw=1.0)
plt.plot(t, ema, 'r-', lw=2.0)
plt.show()'''
