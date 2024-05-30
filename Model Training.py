import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('parkinson_disease.csv')
df.head()

df.shape

df.info()

df.describe()

df = df.groupby('name').mean().reset_index()
df.drop('name', axis=1, inplace=True)

x = df['status'].value_counts()
plt.pie(x.values,
        labels = x.index,
        autopct='%1.1f%%')
plt.show()

# Split features and target
features = df.drop('status', axis=1)
target = df['status']

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)
print(X_train.shape, X_val.shape)

ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
print(X.shape, Y.shape)

from sklearn.metrics import roc_auc_score as ras

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for model in models:
    model.fit(X, Y)
    print(f'{model.__class__.__name__} : ')
    
    # Training AUC
    train_preds = model.predict_proba(X)[:, 1]
    print('Training AUC: ', ras(Y, train_preds))
    
    # Validation AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    print('Validation AUC: ', ras(Y_val, val_preds))
    print()
