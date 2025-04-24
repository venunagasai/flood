#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
import joblib
 
df=pd.read_csv("flood.csv")
df['FloodProbability'] = df['FloodProbability'].apply(lambda x: 1 if x >= 0.5 else 0)
df['FloodProbability'].value_counts()
df['Landslides'].value_counts()
box_plots=[]
for i in df.columns:
    if (i!='Landslides') and (i!='FloodProbability'):
        box_plots.append(i)
for i, column in enumerate(box_plots, 1):
    print(i,column)
plt.figure(figsize=(20, 15))
for i, column in enumerate(box_plots, 1):
    plt.subplot(5, 4, i)
    sns.boxplot(x=df[column])  
    plt.title(column)  
 
plt.tight_layout()
plt.show()
outlier_cols=box_plots
for i in outlier_cols:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    iqr=q3-q1
    print(f"Number of records being dropped from {i} is {len(df)-len(df[(df[i]>q1-1.5*iqr)&(df[i]<q3+1.5*iqr)])}")
    df=df[(df[i]>q1-1.5*iqr)&(df[i]<q3+1.5*iqr)]
 
df_class=df.drop(columns=['Landslides'])
df['FloodProbability'].value_counts()
X=df_class.drop(columns=['FloodProbability'])
Y=df_class['FloodProbability']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
 
lr1=LogisticRegression()
start_time=time.time()
lr1.fit(x_train,y_train)
end_time=time.time()
lrt=end_time-start_time
lr1_pred=lr1.predict(x_test)
print("Accuracy of Logistic Regression:",accuracy_score(lr1_pred,y_test))
print(classification_report(lr1_pred,y_test))
 
dt1=DecisionTreeClassifier()
start_time=time.time()
dt1.fit(x_train,y_train)
end_time=time.time()
dtt=end_time-start_time
dt1_pred=dt1.predict(x_test)
print("Accuracy of Decision Tree:",accuracy_score(dt1_pred,y_test))
print(classification_report(dt1_pred,y_test))
 
rf1=RandomForestClassifier()
start_time=time.time()
rf1.fit(x_train,y_train)
end_time=time.time()
rft=end_time-start_time
rf1_pred=rf1.predict(x_test)
print("Accuracy of Random Forest:",accuracy_score(rf1_pred,y_test))
print(classification_report(rf1_pred,y_test))
 
KNN1=KNeighborsClassifier()
start_time=time.time()
KNN1.fit(x_train,y_train)
end_time=time.time()
KNNt=end_time-start_time
KNN1_pred=KNN1.predict(x_test)
print("Accuracy of KNN:",accuracy_score(KNN1_pred,y_test))
print(classification_report(KNN1_pred,y_test))
 
GNB1=GaussianNB()
start_time=time.time()
GNB1.fit(x_train,y_train)
end_time=time.time()
GNBt=end_time-start_time
GNB1_pred=GNB1.predict(x_test)
print("Accuracy of GNB:",accuracy_score(GNB1_pred,y_test))
print(classification_report(GNB1_pred,y_test))
 
svm1=SVC()
start_time=time.time()
svm1.fit(x_train,y_train)
end_time=time.time()
svmt=end_time-start_time
svm1_pred=svm1.predict(x_test)
print("Accuracy of SVM:",accuracy_score(svm1_pred,y_test))
print(classification_report(svm1_pred,y_test))
 
results_df={"Algorithm":['Logistic Regression',"SVM",'Decision Tree','Random Forest','KNN','GNB'],"Accuracy":[accuracy_score(lr1_pred,y_test),accuracy_score(svm1_pred,y_test),accuracy_score(dt1_pred,y_test),accuracy_score(rf1_pred,y_test),accuracy_score(KNN1_pred,y_test),accuracy_score(GNB1_pred,y_test)],  
            "Recall":[recall_score(lr1_pred,y_test),recall_score(svm1_pred,y_test),recall_score(dt1_pred,y_test),recall_score(rf1_pred,y_test),recall_score(KNN1_pred,y_test),recall_score(GNB1_pred,y_test)],
            "Precision":[precision_score(lr1_pred,y_test),precision_score(svm1_pred,y_test),precision_score(dt1_pred,y_test),precision_score(rf1_pred,y_test),precision_score(KNN1_pred,y_test),precision_score(GNB1_pred,y_test)],
           "F1-Score":[f1_score(lr1_pred,y_test),f1_score(svm1_pred,y_test),f1_score(dt1_pred,y_test),f1_score(rf1_pred,y_test),f1_score(KNN1_pred,y_test),f1_score(GNB1_pred,y_test)]
           }
results_df=pd.DataFrame(results_df)
 
plt.figure(figsize=(25,3))
plt.subplot(1,2,1)
plt.title("Accuracies of algorithms")
plt.xticks(rotation=45)
sns.lineplot(data=results_df,x=results_df['Algorithm'],y=results_df['Accuracy'])
ax=results_df.set_index('Algorithm').plot(kind='bar', figsize=(13, 6))
plt.ylabel('Scores')
plt.title('Algorithm Performance Metrics')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=90)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
 
y_pred_proba = lr1.predict_proba(x_test)[:, 1]
 
def plot_roc_curve(y_test, y_pred_proba):
   
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
 
plot_roc_curve(y_test, y_pred_proba)
 
time_df={"Algorithm":['Logistic Regression',"SVM",'Decision Tree','Random Forest','KNN','GNB'],
         "Execution Time in Seconds":[lrt,svmt,dtt,rft,KNNt,GNBt]}
time_df=pd.DataFrame(time_df)
 
X=df.drop(columns=['Landslides'])
Y=df['Landslides']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
 
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
lrr_pred = linear_reg.predict(x_test)
 
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(x_train, y_train)
rr_pred = ridge_reg.predict(x_test)
 
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x_train, y_train)
lsr_pred = lasso_reg.predict(x_test)
 
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_reg.fit(x_train, y_train)
enr_pred = elastic_net_reg.predict(x_test)
 
xgb_reg = XGBRegressor(n_estimators=100)
xgb_reg.fit(x_train, y_train)
xgb_pred = xgb_reg.predict(x_test)
 
gb_reg = GradientBoostingRegressor(n_estimators=100)
gb_reg.fit(x_train, y_train)
gb_pred = gb_reg.predict(x_test)
 
results_dict = {
    'Algorithm': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'R2': []
}
 
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # manually compute RMSE
    r2 = r2_score(y_test, y_pred)

    results_dict['Algorithm'].append(model_name)
    results_dict['MAE'].append(mae)
    results_dict['MSE'].append(mse)
    results_dict['RMSE'].append(rmse)
    results_dict['R2'].append(r2)

 
evaluate_model(y_test, lrr_pred, "Linear Regression")
evaluate_model(y_test, rr_pred, "Ridge Regression")
evaluate_model(y_test, lsr_pred, "Lasso Regression")
evaluate_model(y_test, enr_pred, "Elastic Net Regression")
evaluate_model(y_test, xgb_pred, "XGBoost Regression")
evaluate_model(y_test, gb_pred, "Gradient Boosting Regression")
results_df=pd.DataFrame(results_dict)
 
ax=results_df.set_index('Algorithm').plot(kind='bar', figsize=(13, 6))
 
def test(test_series,df_class,lr1,linear_reg):
    test=pd.DataFrame(data=[test_series],columns=df_class.columns)
    test.drop(columns=['FloodProbability'],inplace=True)
    a=lr1.predict(test)
    test['FloodProbability']=a
    number_of_ls=linear_reg.predict(test)
    if(a==1):
        print(f"Model predicts that the area has chances of floods and getting {number_of_ls} number of landslides")
    else:
        print(f"Model predicts that the area does not have chances of floods but may face {number_of_ls} number of landslides")
   
    return
 
test(df.iloc[2],df_class,lr1,linear_reg)

# Create the 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save models into the directory
joblib.dump(lr1, "model/logistic_regression_model.pkl")
joblib.dump(linear_reg, "model/linear_regression_model.pkl")