# 策略的框架, 回测的主函数, pro版

# pro版对应指数数据库: stock_index_pro.sql
# pro版对应指数获取: run stock_index_pro.py
# 建立stock_pool数据库: stock_stock_all.sql
# stock_pool采集: run Init_StockAll_Sp.py, 且main中stock_pool需为Init_StockAll_Sp.py中stock_pool的子集
# stock_all表的瘦身版: stock_stock_info.sql, 表结构相同, 但删除了冗余数据, 用于提高回测运行速度

# 模型训练中间集: model_ev_mid.sql, 用于暂存回测时间序列内的部分数据，并用于最终计算F1分值
# 模型训练结果集: model_ev_resu.sql, 用于存储股票在某个时间点上的F1分值
# 策略回测时, 资产账单表:stock_my_capital.sql, 该表内含一条初始数据, 用于定义初始资金(默认为1M RMB), 可根据回测场景自行修改
# 策略回测时, 当前股票资产详情表: stock_my_stock_pool.sql, 主要字段为: 持仓股票代码，成交均价，持仓量，持仓天数

# 导入包
import pymysql
import Model_Evaluate as ev
import Filter
import Portfolio as pf
from pylab import *
import Cap_Update_daily as cap_update
import tushare as ts
import Cal_Quota as cqt

# 定义一个运行时间函数
def Cal_Time_Cost(time_cost):
    mm, ss = divmod(time_cost, 60)
    hh, mm = divmod(mm, 60)
    dd, hh = divmod(hh, 24)

    return dd, hh, mm, ss

# 主函数
if __name__ == '__main__':
    # 计时开始
    time_start = time.time()

    # 建立数据库连接,设置tushare的token,定义一些初始化参数
    db = pymysql.connect(host='localhost', user='root', passwd='your password', db='your dbname', charset='utf8mb4')
    cursor = db.cursor()
    ts.set_token('your token')
    pro = ts.pro_api()

    # 选取回测区间
    year = 2019
    date_seq_start = str(year) + '-07-24'
    date_seq_end = str(year) + '-08-23'

    # 计算一个真实时间间隔, 用于折算Sharp Rate中对应的无风险利率
    dt_start = datetime.datetime.strptime(date_seq_start, '%Y-%m-%d')
    dt_end = datetime.datetime.strptime(date_seq_end, '%Y-%m-%d')
    delta_dt = (dt_end - dt_start).days
    
    # 设定需要进行回测的股票池, 取云计算相关的: 中国软件, 中兴通讯, 浪潮信息, 用友网络, 宝信软件
    # 高端的策略在于选股, 目前没有能力完成
    stock_pool = ['600536.SH', '000063.SZ', '000977.SZ', '600588.SH', '600845.SH']

    # 先清空之前的测试记录, 仅流初始资金1M RMB的1条记录(seq=1), 如需更改, 建议在stock_my_capital.sql中完成
    sql_wash1 = 'delete from my_capital where seq != 1'
    cursor.execute(sql_wash1)
    db.commit()
    sql_wash2 = 'truncate table my_stock_pool'
    cursor.execute(sql_wash2)
    db.commit()

    # 清空行情源表, 并插入相关股票的行情数据, 该操作是为了提高回测计算速度而剔除行情表(stock_all)中的冗余数据
    sql_wash3 = 'truncate table stock_info'
    cursor.execute(sql_wash3)
    db.commit()

    # 清空模型训练结果表, 如果还有在Model_Evaluate中训练的结果, 本小段代码谨慎使用
    sql_wash4 = 'truncate table model_ev_resu'
    cursor.execute(sql_wash4)
    db.commit()

    # 将股票池list转换为一个tuple
    in_str = '('
    for x in range(len(stock_pool)):
        if x != len(stock_pool)-1:
            in_str += str('\'') + str(stock_pool[x])+str('\',')
        else:
            in_str += str('\'') + str(stock_pool[x]) + str('\')')
    sql_insert = "insert into stock_info(select * from stock_all a where a.stock_code in %s)" % in_str
    cursor.execute(sql_insert)
    db.commit()

    # 建回测时间序列
    back_test_date_start = (datetime.datetime.strptime(date_seq_start, '%Y-%m-%d')).strftime('%Y%m%d')
    back_test_date_end = (datetime.datetime.strptime(date_seq_end, "%Y-%m-%d")).strftime('%Y%m%d')
    # 从Tushare中获取交易日历
    df = pro.trade_cal(exchange_id='', is_open=1, start_date=back_test_date_start, end_date=back_test_date_end)
    date_temp = list(df.iloc[:, 1])
    date_seq = [(datetime.datetime.strptime(x, "%Y%m%d")).strftime('%Y-%m-%d') for x in date_temp]
    print(date_seq)

    # 开始模拟交易
    index = 1
    day_index = 0
    for i in range(1, len(date_seq)):
        day_index += 1
        # 每日推进式建模，并获取对下一个交易日的预测结果
        for stock in stock_pool:
            try:
                ans2 = ev.model_eva(stock, date_seq[i], 90, 365)
                # print('Date : ' + str(date_seq[i]) + ' Update : ' + str(stock))
            except Exception as ex:
                print(ex)
                continue

        # 每5个交易日更新一次配仓比例, 简单按照马科维茨理论
        if divmod(day_index + 4, 5)[1] == 0:
            portfolio_pool = stock_pool
            # stock_pool至少需要5只票才可以规划配仓
            if len(portfolio_pool) < 5:
                print('Less than 5 stocks for portfolio!! state_dt : ' + str(date_seq[i]))
                continue
            # 这里取了年份值，既投资组合的回测窗口达到2000+自然日
            pf_src = pf.get_portfolio(portfolio_pool, date_seq[i-1], year)

            # 取最佳收益方向的资产组合
            risk = pf_src[1][0]
            weight = pf_src[1][1]
            # 更新持仓权重, 天数等
            Filter.filter_main(portfolio_pool, date_seq[i], date_seq[i-1], weight)
        else:
            Filter.filter_main([], date_seq[i], date_seq[i - 1], [])
            # 没有交易下的资金跟新, 基于跟新市值变动
            cap_update_ans = cap_update.cap_update_daily(date_seq[i])
        print('Accomplished the Date:  ' + str(date_seq[i]) + '\n')
    print('ALL FINISHED!!!')

    # 参考指数准备, 更改为来自: stock_index_pro
    index_name = 'HS300'
    sql_show_btc = "select * from stock_index_pro a where a.stock_code = '%s' and a.state_dt >= '%s' and a.state_dt <= '%s' order by state_dt asc" %\
                   (index_name, date_seq_start, date_seq_end)
    cursor.execute(sql_show_btc)
    done_set_show_btc = cursor.fetchall()

    # 生成指数的损益率序列
    btc_x = list(range(len(done_set_show_btc)))
    btc_y = [100 * (x[2] / done_set_show_btc[0][2] - 1) for x in done_set_show_btc]
    btc_base = [x[2] for x in done_set_show_btc]

    dict_anti_x = {}
    dict_x = {}
    # 对于回测的日期序列, 建立两份次序字典
    for a in range(len(btc_x)):
        dict_anti_x[btc_x[a]] = done_set_show_btc[a][0]
        dict_x[done_set_show_btc[a][0]] = btc_x[a]

    # 将my_capital表中初始记录的state_dt设置为回测的起始日
    sql_add_base_cap_dt = "update my_capital set state_dt = '%s' where seq = 1" % date_seq_start
    cursor.execute(sql_add_base_cap_dt)

    # my_capital表中, 涵盖了所有回测序列的日期, 但如果某交易日发生超过一笔的交易, 则state_dt会有所重复, 需保留最新记录
    sql_show_profit = "select a.capital, a.state_dt from my_capital a where a.state_dt is not null and" \
                      " a.seq = any(select max(seq) from my_capital group by state_dt)"
    cursor.execute(sql_show_profit)
    done_set_show_profit = cursor.fetchall()
    # 建立策略的损益率序列
    profit_x = [dict_x[x[1]] for x in done_set_show_profit]
    profit_y = [100 * (x[0] / done_set_show_profit[0][0] - 1) for x in done_set_show_profit]

    # 生成资金序列
    capital_y = [x[0] for x in done_set_show_profit]

    # 生成收益率
    print('\nGet Return Rate')
    Exp_Return = cqt.Cal_Return_Rate(capital_y)
    Idx_Return = cqt.Cal_Return_Rate(btc_base)
    print('Strategy: {:.02f}%'.format(Exp_Return[0] * 100))
    print('Base:     {:.02f}%'.format(Idx_Return[0] * 100))

    # 生成回撤序列和最大回撤率
    print('\nGet Max Withdrawal Rate')
    cap_Wdl_Rate = cqt.Cal_Withdrawal_Rate(capital_y)
    btc_Wdl_Rate = cqt.Cal_Withdrawal_Rate(btc_base)
    print('Strategy: {:.02f}%'.format(cap_Wdl_Rate[0] * 100))
    print('Base:     {:.02f}%'.format(btc_Wdl_Rate[0] * 100))

    # 生成夏普率与风险
    print('\nGet Sharp Rate and Risk')
    Stg_Sharp = cqt.Cal_Sharp_Rate(capital_y)
    btc_Sharp = cqt.Cal_Sharp_Rate(btc_base)
    print('Strategy Sharp: {:.04f}'.format(Stg_Sharp[0]), ' Risk:{:.04f}'.format(Stg_Sharp[1]))
    print('Base     Sharp: {:.04f}'.format(btc_Sharp[0]), ' Risk:{:.04f}'.format(btc_Sharp[1]))

    # 生成信息比率
    print('\nGet Information Ratio')
    IR = cqt.Cal_Info_Ratio(capital_y, btc_base)
    print('Information Ratio: {:.04f}'.format(IR[0]), ' Risk:{:.04f}'.format(IR[1]))

    # 可视化模块
    # 统一横坐标为交易日期, 函数没看懂
    def c_fnx(val, poz):
        if val in dict_anti_x.keys():
            return dict_anti_x[val]
        else:
            return ''
    # 生成画布
    fig = plt.figure(figsize=(13, 8))

    # 绘制收益率曲线 (含大盘基准收益曲线)
    ax1 = fig.add_subplot(211)
    ax1.xaxis.set_major_formatter(FuncFormatter(c_fnx))

    l11, = plt.plot(btc_x, btc_y, color='blue')
    l12, = plt.plot(profit_x, profit_y, color='red')

    plt.text(profit_x[-1], profit_y[-1], str(round(profit_y[-1], 2))+'%',
             ha='center', va='bottom', fontsize=12, color='red')
    plt.text(btc_x[-1], btc_y[-1], str(round(btc_y[-1], 2))+'%',
             ha='center', va='top', alpha=0.6, fontsize=12, color='blue')

    plt.text(profit_x[profit_y.index(max(profit_y))], max(profit_y), str(round(max(profit_y), 2)) + '%',
             ha='center', va='bottom', fontsize=12, color='red')
    plt.text(btc_x[btc_y.index(max(btc_y))], max(btc_y), str(round(max(btc_y), 2)) + '%',
             ha='center', va='top', alpha=0.6, fontsize=12, color='blue')

    # 定义指数代码标注的Legends信息
    index_label = {
        'SH': 'Shanghai Composite Index',
        'SZ': 'Shenzhen Component Index',
        'S50': 'SSE 50 Index',
        'HS300': 'CSI 300 Index',
        'ZZ500': 'CS 500 Index',
        'ZX': 'SSE SME Composite Index',
        'CY': 'Growth Enterprise Market Index'
    }

    # 设置Label及Legend
    plt.ylabel('Rate of Return (%)')
    plt.legend(handles=[l11, l12, ], labels=[index_label[index_name], 'The Light of Utech Index'], loc='best')

    ax2 = fig.add_subplot(212)
    ax2.xaxis.set_major_formatter(FuncFormatter(c_fnx))

    l21, = plt.plot(btc_x, - 100 * np.array(btc_Wdl_Rate[2]), color='blue')
    l22, = plt.plot(profit_x, - 100 * np.array(cap_Wdl_Rate[2]), color='red')

    plt.text(profit_x[cap_Wdl_Rate[1]], - 100 * cap_Wdl_Rate[2][cap_Wdl_Rate[1]],  str(round(- 100 * cap_Wdl_Rate[0], 2)) + '%',
             ha='center', va='bottom', fontsize=12, color='red')
    plt.text(btc_x[btc_Wdl_Rate[1]], - 100 * btc_Wdl_Rate[2][btc_Wdl_Rate[1]], str(round(- 100 * btc_Wdl_Rate[0], 2)) + '%',
             ha='center', va='top', alpha=0.6, fontsize=12, color='blue')

    plt.text(profit_x[-1], - 100 * cap_Wdl_Rate[2][-1], str(round(- 100 * cap_Wdl_Rate[2][-1], 2)) + '%',
             ha='center', va='bottom', fontsize=12, color='red')
    plt.text(btc_x[-1],  - 100 * btc_Wdl_Rate[2][-1], str(round(- 100 * btc_Wdl_Rate[2][-1], 2)) + '%',
             ha='center', va='top', alpha=0.6, fontsize=12, color='blue')

    # 设置Label及Legend
    plt.ylabel('Withdrawal Rate (%)')
    plt.legend(handles=[l21, l22, ], labels=[index_label[index_name], 'The Light of Utech Index'], loc='best')

    plt.show()

    # 关闭数据库
    cursor.close()
    db.close()

    # 返回全部回测过程用时
    time_end = time.time()
    time_cost = time_end - time_start
    DHMS = Cal_Time_Cost(time_cost)
    print("\nTime of All Cost: %d Day(s) %02dHour(s) %02dMinute(s) %02d.%02dSecond(s)" %
          (DHMS[0], DHMS[1], DHMS[2], DHMS[3], math.modf(DHMS[3])[0] * 100))
