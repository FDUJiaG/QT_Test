# 数据采集至数据库

import datetime
import tushare as ts
import pymysql

if __name__ == '__main__':

    # 设置tushare pro的token并获取连接, 仅首次和重置时需获取, 对于日K每分钟最多调取两百次
    ts.set_token('your token')
    pro = ts.pro_api()

    # 设定获取日线行情的初始日期和终止日期，暂时将终止日期设定为昨天
    start_dt = '20100101'
    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_dt = time_temp.strftime('%Y%m%d')

    # 建立数据库连接,剔除已入库的部分
    db = pymysql.connect(host='localhost', user='root', passwd='your password', db='your dbname', charset='utf8mb4')
    cursor = db.cursor()

    # 设定需要获取数据的股票池, 取云计算相关的: 中兴通讯, 远光软件, 中国长城, 东方财富, 用友网络, 中科曙光, 中国软件, 浪潮信息, 宝信软件
    stock_pool = ['000063.SZ', '002063.SZ', '000066.SZ', '300059.SZ', '600588.SH', '603019.SH',
                  '600536.SH', '000977.SZ', '600845.SH']
    total = len(stock_pool)
    # 循环获取单个股票的日线行情
    for i in range(len(stock_pool)):
        try:
            df = pro.daily(ts_code=stock_pool[i], start_date=start_dt, end_date=end_dt)

			# 打印进度
            print('Seq: ' + str(i+1) + ' of ' + str(total) + '   Code: ' + str(stock_pool[i]))
            c_len = df.shape[0]
        except Exception as aa:
            print(aa)
            print('No DATA Code: ' + str(i))
            continue
        for j in range(c_len):
            resu0 = list(df.iloc[c_len-1-j])
            resu = []
            for k in range(len(resu0)):
                if str(resu0[k]) == 'nan':
                    resu.append(-8642.97531)
                else:
                    resu.append(resu0[k])
            state_dt = (datetime.datetime.strptime(resu[1], "%Y%m%d")).strftime('%Y-%m-%d')
            try:
                sql_insert = "INSERT INTO stock_all(\
                    state_dt,stock_code,open,close,high,low,vol,amount,pre_close,amt_change,pct_change) VALUES ('%s', '%s', '%.2f', '%.2f','%.2f','%.2f','%i','%.2f','%.2f','%.2f','%.2f')" % (\
                    state_dt, str(resu[0]), float(resu[2]), float(resu[5]), float(resu[3]), float(resu[4]), float(resu[9]), float(resu[10]), float(resu[6]), float(resu[7]), float(resu[8]))
                cursor.execute(sql_insert)
                db.commit()
            except Exception as err:
                continue
    cursor.close()
    db.close()
    print('All Finished!')

''' pro.daily()参数说明

输入参数:
ts_code:    股票代码(二选一)
trade_date: 交易日期(二选一)
start_date: 开始日期(YYYYMMDD)
end_date:   结束日期(YYYYMMDD)

输出参数:
ts_code:    股票代码
trade_date: 交易日期
open:       开盘价
high:       最高价
low:        最低价
close:      收盘价
pre_close:  昨收价
change:     涨跌额
pct_chg:    涨跌幅(未复权)
vol	float:  成交量(手)
amount:     成交额(千元)
'''
