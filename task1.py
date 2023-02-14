import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:/Users/лошара/PycharmProjects/pythonProject1/train.csv")
df_prep = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num

df_prep_num = prepare_num(df_prep)
df_prep_num = df_prep_num.fillna(df_prep_num.median())
scaler = MinMaxScaler()

# отдаем 60% на train, 20% на validate,20% на test
train, validate, test = np.split(df_prep_num.sample(frac=1, random_state=10),
                                 [int(.7 * len(df_prep_num)), int(.8 * len(df_prep_num))])
train_x = train.drop('Survived', axis=1)
train_y = train

validate_x = validate.drop('Survived', axis=1)
validate_y = validate

test_x = test.drop('Survived', axis=1)
test_y = test

# подбор гиперпараметров для RandomForest

crit = ("gini", "entropy")
acc_0 = 0
cr_0 = " "
est_0 = 0
dep = 0
for cr in crit:
    for estimators in range(10, 51):
        for depth in range(1, 11):
            model = RandomForestClassifier(n_estimators=estimators, max_depth=depth, criterion=cr)
            model.fit(train_x, train_y['Survived'])
            predict = model.predict(validate_x)
            acc = accuracy_score(validate_y['Survived'], predict)
            if acc_0 < acc:
                acc_0 = acc
                cr_0 = cr
                est_0 = estimators
                dep = depth
print("RandomForest")
print("criterion: ",cr_0)
print("n_estimators: ",est_0)
print("max_depth: ",dep)
print("accuracy validation: ",acc_0)

model = RandomForestClassifier(n_estimators=est_0, max_depth=dep, criterion=cr_0)
model.fit(train_x, train_y['Survived'])
predict = model.predict(test_x)
acc = accuracy_score(test_y['Survived'], predict)

print("accuracy test: ",acc)
print()

# подбор гиперпараметров для XGBoost
acc_0 = 0
est_0 = 0
dep = 0
for estimators in range(1,20):
  for depth in range(1,10):
    model = XGBClassifier(n_estimators=estimators, max_depth=depth)
    model.fit(train_x, train_y['Survived'])
    predict = model.predict(validate_x)
    acc = accuracy_score(validate_y['Survived'], predict)
    if acc_0 < acc:
      acc_0 = acc
      est_0 = estimators
      dep = depth
print("XGBoost")
print("n_estimators: ",est_0)
print("max_depth: ",dep)
print("accuracy validation: ",acc_0)

model = XGBClassifier(n_estimators=estimators, max_depth=depth)
model.fit(train_x, train_y['Survived'])
predict = model.predict(test_x)
acc = accuracy_score(test_y['Survived'], predict)
print("accuracy test: ",acc)
print()

# подбор гиперпараметров для LogisticRegression
solv = ("lbfgs", "liblinear", "newton-cg", "sag", "saga")
сс = [x/100 for x in range(10, 101, 1)]
c_0 = 0
sol_0 = " "
for c in сс:
  for sol in solv:
    model = LogisticRegression(C=c, solver=sol)
    model.fit(train_x, train_y['Survived'])
    predict = model.predict(validate_x)
    acc = accuracy_score(validate_y['Survived'], predict)
    if acc_0 < acc:
      sol_0 = sol
      c_0 = c
print("LogisticRegression")
print("c: ",c_0)
print("solver: ",sol_0)
print("accuracy validation: ",acc_0)

model = LogisticRegression(C=c, solver=sol)
model.fit(train_x, train_y['Survived'])
predict = model.predict(test_x)
acc = accuracy_score(test_y['Survived'], predict)

print("accuracy test: ",acc)
print()

# метод ближайших соседей
# alg = ("auto", "ball_tree", "kd_tree", "brute")
# alg_0 = " "
# neib_0 = 0
# leaf_0 = 0
# acc_0 = 0
# for neib in range(1,20):
#   for algo in alg:
#     for leaf in range(1,50):

model = KNeighborsClassifier()
model.fit(train_x, train_y['Survived'])
predict = model.predict(validate_x)
acc = accuracy_score(validate_y['Survived'], predict)
      # if acc_0 < acc:
      #   acc_0 = acc
      #   neib_0 = neib
      #   alg_0 = algo
      #   leaf_0 = leaf
# ggggg
# print(alg_0)
# print(neib_0)
# print(leaf_0)
print("KNN")
print("accuracy validation: ",acc)

model = KNeighborsClassifier()
model.fit(train_x, train_y['Survived'])
predict = model.predict(test_x)
acc = accuracy_score(test_y['Survived'], predict)

print("accuracy test: ",acc)

