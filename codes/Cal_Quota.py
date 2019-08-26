# 计算各项量化评估指标
# 包括收益率, 回撤率, 夏普率, 信息比率

# 导入包
import math
import numpy as np

# 计算回报率: 返回基准收益率与年化收益率, 并返回收益率序列, 默认一年250个交易日
def Cal_Return_Rate(seq, yd=250, prt=False):
    seqn = len(seq)
    Return_Rate = (seq[-1] / seq[0]) - 1
    Annual_Rate = math.pow((seq[-1] / seq[0]), yd / seqn) - 1
    Return_List = []
    Base_V = seq[0]
    for i in range(seqn):
        if i == 0:
            Return_List.append(float(0.00))
        else:
            ri = (float(seq[i]) - float(Base_V))/float(Base_V)
            Return_List.append(ri)
    if prt:
        print('Simple Return Rate: {:.02f}%'.format(Return_Rate * 100))
        print('Expected Annual Return Rate: {:.02f}%'.format(Annual_Rate * 100))

    return Return_Rate, Annual_Rate, Return_List

# 计算回撤率: 返回最大回撤率及索引, 回撤率序列
def Cal_Withdrawal_Rate(seq, prt=False):
    Wdl_Rate_List = []
    max_temp = 0
    for i in range(len(seq)):
        max_temp = max(max_temp, seq[i])
        Wdl_Rate = (max_temp - seq[i]) / max_temp
        Wdl_Rate_List.append(round(Wdl_Rate, 4))
        # if prt:
        #     print('No.' + str(i) + ': {:.02f}%'.format(Wdl_Rate * 100))
    Max_Index = Wdl_Rate_List.index(max(Wdl_Rate_List))
    if prt:
        print('Max Withdrawal Rate: {:.02f}%'.format(max(Wdl_Rate_List) * 100), ' Index:', Max_Index)

    return max(Wdl_Rate_List), Max_Index, Wdl_Rate_List

# 计算夏普率: 返回夏普率及风险
def Cal_Sharp_Rate(seq, Rf=0.015, yd=250, prt=False):
    seqn = len(seq)
    seq_return = Cal_Return_Rate(seq, prt=False)
    norisk_return = Rf * seqn / yd
    Risk = float(np.array(seq_return[2]).std())
    Sharp_Rate = (seq_return[0] - norisk_return) / Risk
    if prt:
        print('Sharp Rate: {:.4f}'.format(Sharp_Rate))
        print('Risk: {:.4f}'.format(Risk))

    return Sharp_Rate, Risk

# 计算信息比率: 返回信息比率与跟踪误差
def Cal_Info_Ratio(seq, seq_base, prt=False):
    seq_return = Cal_Return_Rate(seq, prt=False)
    seq_base_return = Cal_Return_Rate(seq_base, prt=False)
    sigma = float((np.array(seq_return[2]) - np.array(seq_base_return[2])).std())
    ir = (seq_return[0] - seq_base_return[0]) / sigma
    if prt:
        print('Information Ratio: {:.4f}'.format(ir))
        print('Tracking Error: {:.4f}'.format(sigma))

    return ir, sigma

if __name__ == '__main__':
    # 给一个测试序列和基准序列
    seq_test = [100, 103, 105, 101, 106, 109, 111, 115, 118, 120, 119, 125, 112, 116, 107, 109, 100, 106, 114, 108]
    seq_base = [100, 101, 103, 102, 105, 107, 119, 113, 114, 115, 113, 116, 110, 111, 105, 107, 102, 104, 108, 105]

    # 测试序列的基本收益率及年化收益率
    print('\nCal Return Rate')
    Return = Cal_Return_Rate(seq_test, prt=True)
    print('Return List:', Return[2])

    # 测试序列的最大回撤率及回撤率列表
    print('\nCal Withdrawal Rate')
    Wdl_Rate = Cal_Withdrawal_Rate(seq_test, True)
    print('Withdrawal Rate List:', Wdl_Rate[2])

    # 测试序列的夏普率与风险
    print('\nCal Sharp Rate')
    Sharp = Cal_Sharp_Rate(seq_test, prt=True)

    # 测试序列基于基准序列的信息比率
    print('\nCal Information Ratio')
    IR = Cal_Info_Ratio(seq_test, seq_base, True)
