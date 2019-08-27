# QT_Test

本测试旨在重现一套比较简单且完备的量化框架，该框架基于现代投资组合理论，并应用主流的机器学习算法（SVM）进行分析。 旨在初步形成一个量化投资的思路，辅助构建科学合理的投资策略

## Anticipate Process

### Preparation

- [SQL Queries](https://github.com/FDUJiaG/QT_Test/tree/master/sqlQueries)
- [Initial Capital](https://github.com/FDUJiaG/QT_Test/blob/master/sqlQueries/my_capital.sql) of Loopback Test (optional, default = 1 M)

### Input

- Stock Pool
- Base Stock Index
- Interval of Loopback Test
- Windows of Preproession (optional, default = 365)
- Windows of Loopback Trainning Test (optional, default = 90)
- Windows of Loopack Portfolio (optional, default = year)
- Change Frequency of Portfolio (optional, default =5)

### Main Program

```shell
$ python Init_StockALL_Sp.py
$ python stock_index_pro.py
$ python main_pro.py
```

### Output

- Daily Trading Data in [Stock Pool](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/stock_all.csv) and [Base Index](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/stock_index_pro.csv)
- [Result](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/model_ev_resu.csv) of SVM Model Evaluation
- The [Capital Situation](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/my_capital.csv) during Loopback Test
- The [Stocks Holding](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/my_stock_pool.csv) in Last Loopback Test Day
- [Effect Index](https://github.com/FDUJiaG/QT_Test/blob/master/imag/LoopBack.png) of Quantization
- <a href="###Return，Withdrawal的可视化">Visualization</a> of Return and Withdrawal

## Dependencies

测试使用的Python版本：3.6.8

测试使用的Anaconda版本：1.9.6

### Installation or Upgrade for Tushare

```shell
$ pip install tushare
$ pip install tushare --upgrade
```

### Import Tushare

```python
import tushare as ts
```

tushare版本需大于1.2.10

### Set Token

```python
ts.set_token('your token')
```

完成调取tushare数据凭证的设置，通常只需要设置一次

### Initialize Pro API

```python
pro = ts.pro_api()
# 或者在初始化中直接设置token
pro = ts.pro_api('your token')
```

### Main Data API

```python
pro.daily()       # 获取日K数据（未赋权）
pro.index_daily() # 获取指数行情
pro.trade_cal()   # 获取交易日历
```

### Package

#### Time Handle

```python
import datetime
```

#### MySql Handle

```python
import pymysql.cursors
import sqlalchemy
```

#### Data Handle

```python
import numpy as np
import pandas as pd
from sklearn import svm
import pylab as *
import math
```

## 设计过程

### 数据采集预处理后建模

- 基于[Tushare](https://tushare.pro/document/1?doc_id=131)进行交易数据采集（[股票](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Init_StockALL_Sp.py)，[指数](https://github.com/FDUJiaG/QT_Test/blob/master/codes/stock_index_pro.py)）
- 简单[数据预处理](https://github.com/FDUJiaG/QT_Test/blob/master/codes/DC.py)，生成训练集
- 利用[SVM](https://blog.csdn.net/b285795298/article/details/81977271)算法进行[建模](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Model_Evaluate.py)，并[预测涨跌](https://github.com/FDUJiaG/QT_Test/blob/master/codes/SVM.py)情况，准备开发择时策略

### 模型评估和仓位管理

- 测试区间内[评估指标](https://blog.csdn.net/zhihua_oba/article/details/78677469)的[计算](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Model_Evaluate.py)，包括：Precision，Recall，F1，Negative_Accuracy等值
-  基于[马科维茨理论](https://mp.weixin.qq.com/s/neCSaWK0c4jzWwCfDVFA6A)的[仓位管理](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Portfolio.py)分配，取**次最小的特征值和特征向量**（最佳收益方向）

### 模拟交易测试及回测

- 模拟交易，包括：获取资金账户[数据](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Deal.py)，执行买卖[操作](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Operator.py)，更新持仓天数及买卖[逻辑](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Filter.py)，更新资产表[数据](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Cap_Update_daily.py)等
- 策略框架下，进行[回测](https://github.com/FDUJiaG/QT_Test/blob/master/codes/main_pro.py)并计时
- 计算并返回量化策略[评估指标](https://www.jianshu.com/p/363aa2dd3441)：Return，Withdrawal，Sharp，Risk，IR及Tracking Error等
- 对于Return，Withdrawal的[可视化展示](###Return，Withdrawal的可视化)

## 详细示例
### 获取数据并存储
#### 股票行情

**数据获取**

默认获取从预设时间（我们定义为2010年第1个交易日）到最邻近交易日，股票池所有交易日的行情数据

注意更改[Init_StockALL_Sp.py](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Init_StockALL_Sp.py)中的股票池

```python
# 设定需要获取数据的股票池, 比如与云计算、软件板块相关的标的
# 中兴通讯, 远光软件, 中国长城, 东方财富, 用友网络, 中科曙光, 中国软件, 浪潮信息, 宝信软件
stock_pool = ['000063.SZ', '002063.SZ', '000066.SZ', '300059.SZ', '600588.SH', '603019.SH', '600536.SH', '000977.SZ', '600845.SH']
```

如果对于股票代码，所属板块，上市状态，上市日期等情况不甚了解，可以优先查询股票的[基本信息](https://tushare.pro/document/2?doc_id=25)

 ```python
# 获取股票基本信息列表
data = pro.stock_basic()
 ```

<p align="left">

 <img src='./imag/Loading_Stock_Data.png' width=260 />

**存储至MySQL**

部分示例 <p align="left">

 <img src='./imag/Stock_Pool_Data.png' width=700 />

**注意**

对于使用[main_pro.py](https://github.com/FDUJiaG/QT_Test/blob/master/codes/main_pro.py)正式回测时，我们选择的股票池必须是前述Init_StockALL_Sp.py中股票的子集，且为了后续构建资产组合中Eig不至于过少，规定股票池的度不小于5

由于我们在分析不同问题时，从Init_StockALL_Sp.py中获取标的的数量较为庞大（比如超过1000只股票），但在回测中标的股票数不可能过于庞大（算力限制，一段时间内主成分也不会过多），比如我们在main_pro.py中的股票池标的数就选为最小的5，在[stock_info](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/stock_info.csv)表中会优先精简上述的[stock_all](https://github.com/FDUJiaG/QT_Test/blob/master/sqlT_to_csv/stock_all.csv)表，既相当于从自己获取的数据库中，抓取回测所需股票池中的标的行情数据，这样会在一定程度上提高查询速度，示例不再赘述

#### 指数行情

**数据获取**

默认获取从预设时间（我们定义为2010年第1个交易日）到最邻近交易日，参考指数所有交易日的行情数据

注意更改[stock_index_pro.py](https://github.com/FDUJiaG/QT_Test/blob/master/codes/stock_index_pro.py)中的基准指数

| 指数名称 | 赋予简称 | 交易所/Tushare编码     |
| -------- | -------- | ---------------------- |
| 上证指数 | SH       | 000001.SH              |
| 深圳成指 | SZ       | 399001.SZ              |
| 上证50   | SH50     | 000016.SH              |
| 沪深300  | HS300    | 000300.SH or 399300.SZ |
| 中证500  | ZZ500    | 000905.SH or 399905.SZ |
| 中小板指 | ZX       | 399005.SZ              |
| 创业板   | CY       | 399006.SZ              |

因为股票池中后续进行回测的股票两市均有，且市值相对较重，所以选择沪深300指数较为合理

```python
df = pro.index_daily(ts_code='000300.SH')
# 统一指数标注并删除原复杂指数代码标注
df['stock_code'] = 'HS300'
```

<p align="left">

 <img src='./imag/Loading_Index_Data.png' width=260 />

**存储至MySQL**

部分示例 <p align="left">

 <img src='./imag/Stock_Index.png' width=555 />

### 单个SVM结果

对于涨跌判断的建模，来自对获取数据的特征，基于SVM做分类问题

```python
from sklearn import svm
import DC
# DC是将原始行情数据划分成SVM训练的各项数据集的预处理类

dc = DC.data_collect(stock, start_date, end_date)
train = dc.data_train           # 训练集
target = dc.data_target         # 目标集
test_case = [dc.test_case]      # 测试集
model = svm.SVC()               # 建模
model.fit(train, target)        # 训练
ans2 = model.predict(test_case) # 预测
```

**运行结果** <p align="left">

 <img src='imag/SVM_ans.png' width=260>

### SVM模型评价

有了单个SVM结果后，就可以通过遍历股票池中的标的，并对比SVM训练时，测试区间中的真实情况给予评价

$$Acc (Precision)= \frac{Tp(预测上涨且正确)}{Tp+Fp(预测上涨实际不上涨)}$$

$$Acc(Recall) = \frac{Tp}{Tp+Fn(预测不上涨但实际上涨)}$$

$$\begin{equation}
F1 =\begin{cases}
\begin{array}{cc}
0, & Precision*Recall=0 \\
\frac{2*Precision*Recall}{Precision + Recall}, & else
\end{array}
\end{cases}
\end{equation}$$

$$ACC\_Neg=\frac{Tn(预测为不上涨且正确)}{Tn+Fn(预测为不上涨但实际上涨)}$$

部分效果如下 <p align="left">

 <img src='imag/Model_Evaluate.png' width=650>

再遍历所有回测区间内的交易日，来给出全部的预测情况，对比回测区间内的真实情况

<p align="left">

 <img src='imag/SVM_Model_Evaluate.png' height=400 />

### 仓位管理

<p align="left">

 <img src='imag/Portfolio.png' height=450 />

### Return，Withdrawal的可视化

<p align="left">

 <img src='imag/LoopBack_190724_190823.png' height=425 />