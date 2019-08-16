from sklearn import svm
import DC

if __name__ == '__main__':
    stock = '000066.SZ'     # 以000066.SZ为例子
    dc = DC.data_collect(stock, '2017-01-01', '2019-08-13')
    train = dc.data_train
    target = dc.data_target
    test_case = [dc.test_case]
    model = svm.SVC()               # 建模
    model.fit(train, target)        # 训练
    ans2 = model.predict(test_case) # 预测

    # 输出对2019-08-14的涨跌预测，1表示涨，0表示不涨。
    print('Forecast:', ans2[0])


