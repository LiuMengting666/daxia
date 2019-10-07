# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:58:21 2019

@author: 10104
"""

import numpy as np
from math import sqrt
##创建一个类KNNClassifier，用来进行knn分类
class KNNClassifier:
    ##初始化一个分类器
    def __init__(self, k):
        assert k>=1, "k必须有效"
        self.k = k
        # 设置私有成员变量_
        self._X_train = None
        self._y_train = None
    ##根据训练数据集X_train和y_train训练kNN分类器
    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "训练集的数据和标签的样本个数必须保持一致"
        assert self.k <= X_train.shape[0], \
            "近邻数k必须小于等于训练集样本个数"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):  
        """给定待测预测数据集X_predict,返回表示X_predict的结果向量集""" 
        assert self._X_train is not None and self._y_train is not None, \
            "训练集数据和测试集数据必须非空"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "测试集的每个数据的维度必须和训练集每个数据的维度一致"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待测数据x,返回x的预测结果集"""
        assert x.shape[0] == self._X_train.shape[1], \
            "测试集的每个数据的维度必须和训练集每个数据的维度一致"
        # 欧拉距离
        distances = [sqrt(np.sum((x_train-x)**2)) 
                    for x_train in self._X_train]
        # 然后按照其索引进行从小到大排序
        nearest = np.argsort(distances)

        # 前k个距离最小的标签的点集
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        
        result = {}
        for color in topK_y:
            if result.get(color)==None:
                result[color]=1
            else:
                result[color]+=1
        if(len(result)==2):
            if result[0.0]>result[1.0]:
                return 0
            else:
                return 1
        a = list(result.keys())
        return a[0]
##导入数据
def init_data():
    data = np.loadtxt('HTRU_2_train.csv',delimiter=',')
    data2 = np.loadtxt('HTRU_2_test.csv',delimiter=',')
    dataMatIn,classLabels = np.split(data, (2,), axis=1)#将数据和类别标号
    classLabels=classLabels.ravel()
    return dataMatIn,classLabels,data2

X_train,y_train,X_test= init_data()
clf = KNNClassifier(36)
clf.fit(X_train, y_train)        
y_test_pred = clf.predict(X_test)    
pre = [[] for i in range(700)]
count = 1
for i in y_test_pred:  
        pre[count-1].append(count)
        pre[count-1].append(i)
        count+=1
print(pre)
f = open(r'C:\Users\10104\Desktop\\hahahaha.csv','w')
np.savetxt(r'C:\Users\10104\Desktop\\hahahaha.csv',pre,delimiter=',')
f.close()
