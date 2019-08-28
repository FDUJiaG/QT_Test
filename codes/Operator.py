# 模拟交易：封装函数，用于模拟交易过程中执行买和卖操作

import pymysql.cursors
import Deal

def buy(stock_code, opdate, buy_money):
    # 建立数据库连接
    db = pymysql.connect(host='localhost', user='root', passwd='your password', db='your dbname', charset='utf8mb4')
    cursor = db.cursor()

    deal = Deal.Deal(opdate)
    init_price = deal.stock_map1[stock_code]
    hold_vol = deal.stock_map2[stock_code]
    hold_days = deal.stock_map3[stock_code]
    sql_sell_select = "select * from stock_info a where a.state_dt = '%s' and a.stock_code = '%s'" % (opdate, stock_code)
    cursor.execute(sql_sell_select)
    done_set_sell_select = cursor.fetchall()
    if len(done_set_sell_select) == 0:
        return -1
    sell_price = float(done_set_sell_select[0][3])

    # 市值增幅超过4%止盈
    # 平仓手续费约为0.16%
    if sell_price > init_price * 1.04 and hold_vol > 0:
        new_money_lock = deal.cur_money_lock - sell_price * hold_vol
        new_money_rest = deal.cur_money_rest + sell_price * hold_vol * 0.9984
        new_capital = deal.cur_capital + (sell_price * 0.9984 - init_price) * hold_vol
        new_profit = (sell_price * 0.9984 - init_price) * hold_vol
        new_profit_rate = sell_price * 0.9984 / (init_price * 1.0005)
        sql_sell_insert = "insert into my_capital(capital,money_lock,money_rest,deal_action,stock_code,stock_vol,profit,profit_rate,bz,state_dt,deal_price)values('%.02f','%.2f','%.02f','%s','%s','%d','%.2f','%.04f','%s','%s','%.02f')" %\
                          (new_capital, new_money_lock, new_money_rest, 'SELL', stock_code, hold_vol, new_profit, new_profit_rate, 'GOODSELL', opdate, sell_price)
        cursor.execute(sql_sell_insert)
        db.commit()

        sql_sell_update = "delete from my_stock_pool where stock_code = '%s'" % (stock_code)
        cursor.execute(sql_sell_update)
        db.commit()
        db.close()
        return 1

    # 市值跌幅超过3%止损
    elif sell_price < init_price * 0.97 and hold_vol > 0:
        new_money_lock = deal.cur_money_lock - sell_price * hold_vol
        new_money_rest = deal.cur_money_rest + sell_price * hold_vol * 0.9984
        new_capital = deal.cur_capital + (sell_price * 0.9984 - init_price) * hold_vol
        new_profit = (sell_price * 0.9984 - init_price) * hold_vol
        new_profit_rate = sell_price * 0.9984 / (init_price * 1.0005)
        sql_sell_insert2 = "insert into my_capital(capital,money_lock,money_rest,deal_action,stock_code,stock_vol,profit,profit_rate,bz,state_dt,deal_price)values('%.02f','%.2f','%.02f','%s','%s','%d','%.2f','%.04f','%s','%s','%.02f')" %\
                           (new_capital, new_money_lock, new_money_rest, 'SELL', stock_code, hold_vol, new_profit, new_profit_rate, 'BADSELL', opdate, sell_price)
        cursor.execute(sql_sell_insert2)
        db.commit()
        sql_sell_update2 = "delete from my_stock_pool where stock_code = '%s'" % (stock_code)
        cursor.execute(sql_sell_update2)
        db.commit()

        db.close()
        return 1

    # 持仓超过4个交易日强行平仓
    elif hold_days >= 4 and hold_vol > 0:
        new_money_lock = deal.cur_money_lock - sell_price * hold_vol
        new_money_rest = deal.cur_money_rest + sell_price * hold_vol * 0.9984
        new_capital = deal.cur_capital + (sell_price * 0.9984 - init_price) * hold_vol
        new_profit = (sell_price * 0.9984 - init_price) * hold_vol
        new_profit_rate = sell_price * 0.9984 / (init_price * 1.0005)
        sql_sell_insert3 = "insert into my_capital(capital,money_lock,money_rest,deal_action,stock_code,stock_vol,profit,profit_rate,bz,state_dt,deal_price)values('%.02f','%.2f','%.02f','%s','%s','%d','%.2f','%.04f','%s','%s','%.02f')" %\
                           (new_capital, new_money_lock, new_money_rest, 'OVERTIME', stock_code, hold_vol, new_profit, new_profit_rate, 'OVERTIMESELL', opdate, sell_price)
        cursor.execute(sql_sell_insert3)
        db.commit()
        sql_sell_update3 = "delete from my_stock_pool where stock_code = '%s'" % (stock_code)
        cursor.execute(sql_sell_update3)
        db.commit()
        db.close()
        return 1

    # 如果预测下跌，则卖出
    elif predict == -1 and hold_vol > 0:
        new_money_lock = deal.cur_money_lock - sell_price * hold_vol
        new_money_rest = deal.cur_money_rest + sell_price * hold_vol * 0.9984
        new_capital = deal.cur_capital + (sell_price * 0.9984 - init_price) * hold_vol
        new_profit = (sell_price * 0.9984 - init_price) * hold_vol
        new_profit_rate = sell_price * 0.9984 / (init_price * 1.0005)
        sql_sell_insert4 = "insert into my_capital(capital,money_lock,money_rest,deal_action,stock_code,stock_vol,profit,profit_rate,bz,state_dt,deal_price)values('%.02f','%.2f','%.02f','%s','%s','%d','%.2f','%.04f','%s','%s','%.02f')" % (
        new_capital, new_money_lock, new_money_rest, 'Predict', stock_code, hold_vol, new_profit, new_profit_rate, 'PredictSell', opdate, sell_price)
        cursor.execute(sql_sell_insert4)
        db.commit()
        sql_sell_update3 = "delete from my_stock_pool where stock_code = '%s'" % (stock_code)
        cursor.execute(sql_sell_update3)
        db.commit()
        db.close()
        return 1
    db.close()
    return 0
