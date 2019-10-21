#(1018 homework)人工智慧
#KNN、CART、LR,比較K值

import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import model_selection  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix #计算混淆矩阵，主要来评估分类的准确性
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

#download dataset
dataset=pd.read_csv("avocado.csv")

# Split out validation dataset
array = dataset.values #將數據庫->數組
X = array[:,1:7] 
Y = array[:,10] #取最後
validation_size = 0.20 #驗證集規模
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #分割数据集

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = [] #建立列表
models.append(('LR', LogisticRegression())) #往models添加元组（算法名稱，算法函数）
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models: #將算法名稱與算法函數分別讀取
	kfold = model_selection.KFold(n_splits=10, random_state=seed) #建立10倍交叉驗證
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) #每一个算法模型作為其中的參數，算每一模型的精度得分
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
	print(msg)
#CART準度最優


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111) #1*1網格 第一子圖
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation) #預測驗證集
print("驗證集精度得分:\n",accuracy_score(Y_validation, predictions))
print("混淆矩陣: \n",confusion_matrix(Y_validation, predictions)) 
print("分類預測報告:\n",classification_report(Y_validation, predictions)) 
#recall=TP/TP+FN
#f1-score=2*accuracy*recall/accuracy+recall


# 對k取1-30的值，計算每个k對應的平均scores
k_range = range(1, 31)
k_scores = []   #k_class is list
for k in k_range:
    knn = KNeighborsClassifier(k)
    # 學習方法為knn，數據分成5分（cv），打分方法為accuracy, 輸出為5為元組
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
 
# 可視化，k值value和accuracy 的關係圖
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross_Validation Accuracy')
plt.show()


