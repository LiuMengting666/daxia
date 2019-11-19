# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:09:13 2019

@author: 10104
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:27:57 2019

@author: 10104
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:10:25 2019

@author: 10104
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:08:35 2019

@author: 10104
"""
from numpy import NaN
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def init_data():
    data = np.loadtxt(r'C:\Users\10104\Desktop\\mean_train.csv',delimiter=',')
    data2 = np.loadtxt(r'C:\Users\10104\Desktop\\mean_test.csv',delimiter=',')
    dataMatIn,classLabels = np.split(data, (13,), axis=1)#将数据和类别标号
    classLabels=classLabels.ravel()
    return dataMatIn,classLabels,data2

#这是对确实数据的处理
# =============================================================================
# train_df = pd.read_csv('train.csv',header=None)
# test_df = pd.read_csv('test.csv',header=None)
# a = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# for j in range(14):
#     for i in range(7194):
#         if(train_df.iloc[i][j]=="?"):
#             train_df.iloc[i,j]=NaN
#     train_df.iloc[:,j]=train_df.iloc[:,j].dropna().astype(int)
#     a[j] = train_df.loc[:,j].dropna().mean()
#     train_df.iloc[:,j] = train_df.iloc[:,j].fillna(a[j])
#     for k in range(1798):
#         if(j!=13 and test_df.iloc[k][j]=="?"):
#             test_df.iloc[k,j]=NaN
#             print(j)
#     if(j!=13):
#         test_df.iloc[:,j] = test_df.iloc[:,j].fillna(a[j])
# f = open(r'C:\Users\10104\Desktop\\mean_train.csv','w')
# f2 = open(r'C:\Users\10104\Desktop\\mean_test.csv','w')
# train_df.to_csv(r'C:\Users\10104\Desktop\\mean_train.csv', sep=',',header=None,index=None)
# test_df.to_csv(r'C:\Users\10104\Desktop\\mean_test.csv', sep=',',header=None,index=None)
# f.close()
# f2.close()
# =============================================================================



X_train,y_train,data2= init_data()

#AX_train = preprocessing.scale(X_train)




#xun = [i/10.0 for i in range(5,21)]
X_train,X_test,y_train,y_test=model_selection.train_test_split(X_train,y_train,test_size=0.20,random_state=5)
h = .02


X_train[:,3] = X_train[:,3]*1.6
X_train[:,0] = X_train[:,0]*2
X_train[:,1] = X_train[:,1]*1.2
X_train[:,4] = X_train[:,4]*1.1
X_train[:,5] = X_train[:,5]*0.5



data2[:,3] = data2[:,3]*1.6
data2[:,0] = data2[:,0]*2
data2[:,1] = data2[:,1]*1.2
data2[:,4] = data2[:,4]*1.1
data2[:,5] = data2[:,5]*0.5
daye = np.copy(X_train[:,7])
#近邻数K的选择------组合1--------------------------------------
kfold = KFold(n_splits = 5,shuffle = False)
xun = [i/10.0 for i in range(5,21)]
for i in xun:
    X_train[:,7] = daye

    X_train[:,7] = X_train[:,7]*i
    accuracy = []
    for n_neighbors in range(70,100):                 
        for weights in ['uniform']:
            temp_wrong = []
            for train_index,test_index in kfold.split(X_train):
                clf = KNeighborsClassifier(n_neighbors)
                clf.fit(X_train[train_index],y_train[train_index])
                pred = clf.predict(X_train[test_index])
                temp_wrong.append(1-accuracy_score(y_train[test_index],pred))
            accuracy.append(temp_wrong)
    accuracy = np.array(accuracy)
    mean = np.mean(accuracy,axis = 1)
    
    std = np.std(accuracy,axis = 1)
    b = mean
    min_acc = np.min(b)
    index_min = 70 + np.argmin(b)
    print(i,index_min,1-min_acc)


clf = neighbors.KNeighborsClassifier(90, weights=weights)
clf.fit(X_train,y_train)        
y_test_pred = clf.predict(data2)    
print(y_test_pred.shape)

pre = [[] for i in range(1798)]
count = 1
for i in y_test_pred:  
     pre[count-1].append(count)
     pre[count-1].append(i)
     count+=1
    
f = open(r'C:\Users\10104\Desktop\\la.csv','w')
np.savetxt(r'C:\Users\10104\Desktop\\la.csv',pre,delimiter=',')
f.close()



    
