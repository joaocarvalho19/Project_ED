#numeric: pandas and numpy
import numpy as np
import pandas as pd

from sklearn import linear_model, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


df = pd.read_csv('data/sensoringData_feature_prepared_20_19.0_2.csv',header = 0)

# id is useless
df.drop('id',axis=1,inplace=True)
df.drop('user',axis=1,inplace=True)
df.drop('timestamp',axis=1,inplace=True)

feature_list = list(df.columns[:-2])
print(len(feature_list))
#print(df.head())

# print the number of missing 
#df.isnull().sum()


print(df['activity'].unique())

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
df['activity']= label_encoder.fit_transform(df['activity'])
 
print(df['activity'].unique())

y=df.values[:,-1]
#Y = np.array(y).astype(int)
#print(y)

X=df.values[:,0:-2]
print(f"Features: {len(X[0])}")
print(f"Examples: {len(X)}")


def get_best_x_features(X, y, num_features=50):
    #df = (df - np.min(df))/(np.max(df) - np.min(df))
    
    k_bestfeatures = SelectKBest(score_func = f_classif, k=num_features)
    k_bestfeatures.fit(X, y)
    
    # what are scores for the features
    #for i in range(len(rankings.scores_)):
        #print('Feature %d: %f' % (i, rankings.scores_[i]))
    
    # transform train input data
    X = k_bestfeatures.transform(X)
    return X

    b=list(rankings.scores_)
    a=list(range(0,len(b)))

    sf = [g for _,g in sorted(zip(b,a))]
    sf=sf[len(a)-num_features:len(a)]
    sf=reversed(sf)
    inx=[]
    for chosen in sf:
        inx.append(chosen)
    
    dataset = pd.DataFrame(df, dtype='float')
    
    return dataset[inx].to_numpy(dtype='float')

X_50 = get_best_x_features(X,y,50)
print(f"Features: {len(X_50[0])}")
print(f"Examples: {len(X_50)}")

X_train, X_test, y_train, y_test = train_test_split(X_50, y, test_size=0.3,random_state=109) # 70% training and 30% test
print(y_train)

def knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(X_train,y_train)

    #Predict Output
    y_pred= model.predict(X_test) # 0:Overcast, 2:Mild
    print(y_pred)
    return y_pred

knn_y_pred = knn(X_train, y_train)
print("Accuracy:",accuracy_score(y_test, knn_y_pred))