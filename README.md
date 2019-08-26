# QT_Test

主要为了重现一套比较简单且完备的量化框架，该框架基于现代投资组合理论，并应用主流的机器学习算法进行分析。 旨在初步形成一个量化投资的思路，辅助构建科学合理的投资策略。

## 预期效果

### Preparation

- SQL表

### Input

- Stock Pool
- Base Stock Index

### Main Program

### Output

## 环境准备​

建议使用Python版本：3.6及以上

### 安装及升级tushare

```shell
pip install tushare
pip install tushare --upgrade
```

### 导入tushare

```python
import tushare as ts
```

tushare版本需大于1.2.10

### 设置token

```python
ts.set_token('your token')
```

完成调取tushare数据凭证的设置，通常只需要设置一次

### 初始化pro接口

```python
pro = ts.pro_api()
# 或者在初始化中直接设置token
pro = ts.pro_api('your token')
```

### 主要接口函数

```python
pro.daily()       # 获取日K数据（未赋权）
pro.index_daily()	# 获取指数行情
pro.trade_cal()   # 获取交易日历
```

### 主要包

#### 时间处理

```python
import datetime
```

#### MySql相关

```python
import pymysql.cursors
import sqlalchemy
```

#### 数据处理

```python
import numpy as np
import pandas as pd
from sklearn import svm
import pylab as *
import math
```

## 设计过程

###数据采集预处理后建模

- 基于[Tushare](https://tushare.pro/document/1?doc_id=131)进行交易数据采集（[股票](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Init_StockALL_Sp.py)，[指数](https://github.com/FDUJiaG/QT_Test/blob/master/codes/stock_index_pro.py)）
- 简单[数据预处理](https://github.com/FDUJiaG/QT_Test/blob/master/codes/DC.py)，生成训练集
- 利用[SVM](https://blog.csdn.net/b285795298/article/details/81977271)算法进行[建模](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Model_Evaluate.py)，并[预测涨跌](https://github.com/FDUJiaG/QT_Test/blob/master/codes/SVM.py)情况，准备开发择时策略

###模型评估和仓位管理

- 测试区间内[评估指标](https://blog.csdn.net/zhihua_oba/article/details/78677469)的[计算](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Model_Evaluate.py)，包括：Precision，Recall，F1，Negative_Accuracy等值
-  基于[马科维茨理论](https://mp.weixin.qq.com/s/neCSaWK0c4jzWwCfDVFA6A)的[仓位管理](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Portfolio.py)分配，取**次最小的特征值和特征向量**（最佳收益方向）

###模拟交易测试及回测

- 模拟交易，包括：获取资金账户[数据](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Deal.py)，执行买卖[操作](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Operator.py)，更新持仓天数及买卖[逻辑](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Filter.py)，更新资产表[数据](https://github.com/FDUJiaG/QT_Test/blob/master/codes/Cap_Update_daily.py)等
- 策略框架下，进行[回测](https://github.com/FDUJiaG/QT_Test/blob/master/codes/main_pro.py)并计时
- 计算并返回量化策略[评估指标](https://www.jianshu.com/p/363aa2dd3441)：Return，Withdrawal，Sharp，Risk，IR及Tracking Error等
- 对于Return，Withdrawal的可视化展示

## 示例
### 数据下载
<img src='imag/Loading_Data.png' align=left height="200">

### 存储到MySQL
![](./imag/Stock_Pool_Data.png)

### 单个SVM结果

<img src='imag/SVM_ans.png' style="float:left;zoom:60%">

### SVM模型评价存储

<img src='imag/SVM_Model_Evaluate.png' style="float:left;zoom:50%">

### 仓位管理

<img src='imag/Portfolio.png' style="float:left;zoom:60%">