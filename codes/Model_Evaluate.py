# 模型评估：通过回测+推进式建模的方式对模型进行评估，主要计算查准率Precision，查全率Recall，F1分值，并存入结果表

from sklearn import svm
import pymysql.cursors
import datetime
import DC
import tushare as ts


def model_eva(stock, state_dt, para_window, para_dc_window):
    # 建立数据库连接，设置tushare token
    db = pymysql.connect(host='localhost', user='root', passwd='your password', db='your dbname', charset='utf8mb4')
    cursor = db.cursor()
    ts.set_token('your token')
    pro = ts.pro_api()

    # 建评估时间序列, para_window参数代表回测窗口长度
    model_test_date_start = (datetime.datetime.strptime(state_dt, '%Y-%m-%d') -
                             datetime.timedelta(days=para_window)).strftime('%Y%m%d')
    model_test_date_end = datetime.datetime.strptime(state_dt, '%Y-%m-%d').strftime('%Y%m%d')
    # 获取交易日历
    df = pro.trade_cal(exchange_id='', is_open=1, start_date=model_test_date_start, end_date=model_test_date_end)
    date_temp = list(df.iloc[:, 1])
    model_test_date_seq = [(datetime.datetime.strptime(x, "%Y%m%d")).strftime('%Y-%m-%d') for x in date_temp]

    # 清空评估用的中间表model_ev_mid
    sql_truncate_model_test = 'truncate table model_ev_mid'
    cursor.execute(sql_truncate_model_test)
    db.commit()

    # # 是否清除结果表
    # if input('Truncate the model_ev_resu table? Press y/[n]') == 'y':
    #     sql_truncate_model_resu = 'truncate table model_ev_resu'
    #     cursor.execute(sql_truncate_model_resu)
    #     db.commit()

    return_flag = 0
    # 开始回测，其中para_dc_window参数代表建模时数据预处理所需的时间窗长度
    for d in range(len(model_test_date_seq)):
        model_test_new_start = (datetime.datetime.strptime(model_test_date_seq[d], '%Y-%m-%d') - datetime.timedelta(
            days=para_dc_window)).strftime('%Y-%m-%d')
        model_test_new_end = model_test_date_seq[d]
        try:
            dc = DC.data_collect(stock, model_test_new_start, model_test_new_end)
            if len(set(dc.data_target)) <= 1:
                continue
        except Exception as exp:
            print("DC Errrrr")
            return_flag = 1
            break

        # 训练模块
        method = 'SVM'
        train = dc.data_train
        target = dc.data_target
        test_case = [dc.test_case]
        model = svm.SVC()  # 建模
        model.fit(train, target)  # 训练
        ans2 = model.predict(test_case)  # 预测

        # 将预测结果插入到中间表
        sql_insert = "insert into model_ev_mid(state_dt,stock_code,resu_predict)values('%s','%s','%.2f')" % (
        model_test_new_end, stock, float(ans2[0]))
        cursor.execute(sql_insert)
        db.commit()
    # 如果dc无法获取, 则所有计算值被赋予0，例如股票上市日期不久
    if return_flag == 1:
        acc = recall = acc_neg = f1 = 0
        return -1
    else:
        # 在中间表中刷真实值
        for i in range(len(model_test_date_seq)):
            sql_select = "select * from stock_all a where " \
                         "a.stock_code = '%s' and a.state_dt >= '%s' order by a.state_dt asc limit 2" % (
                stock, model_test_date_seq[i])
            cursor.execute(sql_select)
            done_set2 = cursor.fetchall()
            if len(done_set2) <= 1:
                break
            resu = -1
            if float(done_set2[1][3]) / float(done_set2[0][3]) > 1.00:
                resu = 1
            sql_update = \
                "update model_ev_mid w set w.resu_real = '%.2f' where w.state_dt = '%s' and w.stock_code = '%s'" % (
            resu, model_test_date_seq[i], stock)
            cursor.execute(sql_update)
            db.commit()

        # 计算查全率
        sql_resu_recall_son = "select count(*) from model_ev_mid a where" \
                              " a.resu_real is not null and a.resu_predict = 1 and a.resu_real = 1"
        cursor.execute(sql_resu_recall_son)
        recall_son = cursor.fetchall()[0][0]
        sql_resu_recall_mon = "select count(*) from model_ev_mid a where a.resu_real is not null and a.resu_real = 1"
        cursor.execute(sql_resu_recall_mon)
        recall_mon = cursor.fetchall()[0][0]
        # 如果统计区间没有实际上涨的情况, 所有指标被赋予0
        if recall_mon == 0:
            acc = recall = acc_neg = f1 = 0
        else:
            recall = recall_son / recall_mon

        # 计算查准率
        sql_resu_acc_son = "select count(*) from model_ev_mid a where" \
                           " a.resu_real is not null and a.resu_predict = 1 and a.resu_real = 1"
        cursor.execute(sql_resu_acc_son)
        acc_son = cursor.fetchall()[0][0]
        sql_resu_acc_mon = "select count(*) from model_ev_mid a where a.resu_real is not null and a.resu_predict = 1"
        cursor.execute(sql_resu_acc_mon)
        acc_mon = cursor.fetchall()[0][0]
        # 如果没有预测上涨的情况，所有指标被赋予0
        if acc_mon == 0:
            acc = recall = acc_neg = f1 = 0
        else:
            acc = acc_son / acc_mon

        # 计算查准率(负样本)
        sql_resu_acc_neg_son = "select count(*) from model_ev_mid a where" \
                               " a.resu_real is not null and a.resu_predict = -1 and a.resu_real = -1"
        cursor.execute(sql_resu_acc_neg_son)
        acc_neg_son = cursor.fetchall()[0][0]
        sql_resu_acc_neg_mon = \
            "select count(*) from model_ev_mid a where a.resu_real is not null and a.resu_predict = -1"
        cursor.execute(sql_resu_acc_neg_mon)
        acc_neg_mon = cursor.fetchall()[0][0]
        if acc_neg_mon == 0:
            acc_neg = -1
        else:
            acc_neg = acc_neg_son / acc_neg_mon

        # 计算 F1 分值
        if acc + recall == 0:
            acc = recall = acc_neg = f1 = 0
        else:
            f1 = (2 * acc * recall) / (acc + recall)

    sql_predict = "select resu_predict from model_ev_mid a where a.state_dt = '%s'" % (model_test_date_seq[-1])
    cursor.execute(sql_predict)
    done_predict = cursor.fetchall()
    predict = 0
    if len(done_predict) != 0:
        predict = int(done_predict[0][0])

    # 将评估结果存入结果表model_ev_resu中
    sql_final_insert = "insert into model_ev_resu(state_dt,stock_code,acc,recall,f1,acc_neg,bz,predict) values ('%s','%s','%.4f','%.4f','%.4f','%.4f','%s','%s')" % (
        state_dt, stock, acc, recall, f1, acc_neg, method, str(predict))
    cursor.execute(sql_final_insert)
    db.commit()
    db.close()

    # 打印结果
    print(str(state_dt), '  Code:' + str(stock), '  Precision: {:.04f}'.format(acc), '  Recall: {:.04f}'.format(recall),
          '  F1: {:.04f}'.format(f1), '  Acc_Neg: {:.04f}'.format(acc_neg))
    return 1

if __name__ == '__main__':
    stock_pool = ['600536.SH', '000063.SZ', '000977.SZ', '600588.SH', '600845.SH']
    for stock in stock_pool:
        ans = model_eva(stock, '2019-08-23', 90, 365)
    print('All Finished !!!')

''' pro.trade_cal()参数说明

输入参数
exchange:	    交易所 SSE上交所 SZSE深交所
start_date:     开始日期
end_date:       结束日期
is_open:        是否交易 '0'休市 '1'交易

输出参数
exchange:       交易所 SSE上交所 SZSE深交所
cal_date:       日历日期
is_open:        是否交易 0休市 1交易
pretrade_date:  上一个交易日
'''
