# QT_Test

主要为了重现一套比较简单且完备的量化框架，该框架基于现代投资组合理论，并应用主流的机器学习算法进行分析。 旨在初步形成一个量化投资的思路，辅助构建科学合理的投资策略。

## 主要过程

- 数据采集预处理后建模
  - 数据采集
  - 简单数据预处理，生成训练集
  - 利用SVM算法进行建模

- 模型简单评估和仓位管理（完成部分）
  - Precision, Recall, F1值计算
  -  仓位管理

- 模拟交易测试及回测（未开始）

## 主要包
```python
import datetime
import tushare as ts
import pymysql.cursors
from sklearn import svm
import numpy as np
import pandas as pd
import sqlalchemy
```

## 主要接口函数
```python
pro.daily()       # 获取日K数据（未赋权）
pro.trade_cal()   # 获取交易日历
```

<img src="https://github.com/FDUJiaG/QT_Test/tree/master/imag/Loading_Data.png">
