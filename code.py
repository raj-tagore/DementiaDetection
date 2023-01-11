#import packages-----------------------------------------------
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier

#1.data slection---------------------------------------------------
dataframe=pd.read_csv("dataset2.csv")

#2.pre processing--------------------------------------------------
#checking  missing values 

#replace the missing values by 0
median = dataframe['MMSE'].median()
dataframe['MMSE'].fillna(median, inplace=True)
median = dataframe['SES'].median()
dataframe['SES'].fillna(median, inplace=True)

#.visulaization---------------------------------------------------
dataframe['Group'] = dataframe['Group'].replace(['Converted'], ['Demented'])

#sns.countplot(x='Group', data=dataframe)

#label encoding
#Encode columns into numeric
label_encoder = preprocessing.LabelEncoder()
dataframe['Group']= label_encoder.fit_transform(dataframe['Group']) 
dataframe['M/F']= label_encoder.fit_transform(dataframe['M/F'])
dataframe['Hand'] = label_encoder.fit_transform(dataframe['Hand'])

sns.countplot(x='Group', data=dataframe)

correlation = dataframe.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')
plt.show()

#3.data splitting--------------------------------------------------
feature_col_names = ["EDUC", "ASF", "eTIV", "MMSE","nWBV"]
predicted_class_names = ['Group']

X = dataframe[feature_col_names].values
y = dataframe[predicted_class_names].values

# spliting the x and y into test and train
test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                    random_state=2)


y_train = y_train.reshape(int(dataframe.shape[0]*(1-test_size)))

# single -- function(model) --> pred
# double -- function(model1, model2) --> pred
# stack -- function([model1, model2... modeln]) --> pred
# get_accuracy -- function(pred) --> acc, pre, sen, spe
        
def get_accuracy(info):
    cm1=confusion_matrix(y_test, info[0]) # [[0, 0, 1...], "LogisticRegression"] or [pred, name]
    # [[34, 37,
    #  [45, 45]]
    TP = cm1[0][0] #true positive
    FP = cm1[0][1]
    FN = cm1[1][0]
    TN = cm1[1][1]
    Total=TP+TN+FP+FN
    accuracy=((TP+TN)/Total)*100
    precision=TP/(TP+FP)*100
    sensitivity=TP/(TP+FN)*100
    specificity = (TN / (TN+FP))*100
    F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    
    print("-----------------------------------------------------")
    print("Performance Metrics for "+ info[1])
    print("Accuracy: ",accuracy,'%')
    print("Precision: ",precision,'%')
    print("Sensitivity: ",sensitivity,'%')
    print("Specificity: ",specificity,'%')
    print("F1_score: ",F1_score,'%')
    
    return [accuracy, precision, sensitivity, specificity, F1_score]


best_score = 0  
best_model_name = None
best_mtype = None

def get_best(otp, name, mtype):
    global best_score
    global best_model_name
    global best_mtype
    metric = otp[4] # metric = f1_score 
    if metric>=best_score:
        best_model_name = name
        best_score = metric
        best_mtype = mtype
        return best_model_name, best_score, mtype

def single_model(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    info = [pred, model.__class__.__name__]
    otp = get_accuracy(info)
    get_best(otp, info[1], 'single')
    return otp
    
def double_model(model1, model2):
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_train) 
    model2.fit(X_train, pred1)
    pred = model2.predict(X_test)
    info = [pred, f'{model1.__class__.__name__}, {model2.__class__.__name__}']
    otp = get_accuracy(info)
    get_best(otp, info[1], 'double')
    return otp

def stacked_model(layer):
    # generate estimator list
    estimator_list = []
    model_stack = f''
    for i in layer:
        estimator_list.append((i.__class__.__name__, i))
        model_stack+=i.__class__.__name__+', '
    stack_model = StackingClassifier(
        estimators=estimator_list, 
        final_estimator= LogisticRegression())
    stack_model.fit(X_train, y_train)
    pred = stack_model.predict(X_test)
    info = [pred, model_stack]
    otp = get_accuracy(info)
    get_best(otp, info[1], 'stacked')
    return otp
        
# initialize all models

svm = SVC(kernel="linear", C=0.1,random_state=0)
lr = LogisticRegression(random_state = 0, max_iter=2000)  
rf = RandomForestClassifier(n_estimators=100)
nb = GaussianNB()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='error')
DT = DecisionTreeClassifier()

all_models = [DT, svm, rf, nb, xgb, lr]
all_modelnames = {i.__class__.__name__: i for i in all_models}

data = [[]]
for i in all_models:
    x = single_model(i)
    data.append([i.__class__.__name__, x[4]])

for i in all_models:
    for j in all_models:
        x = double_model(i, j)
        data.append([i.__class__.__name__+j.__class__.__name__, x[4]])
  
layer_stack = [svm, rf, nb, xgb, DT, lr]      
stacked_model(layer_stack)

df69 = pd.DataFrame(data, columns = ['Name', 'f1-score'])
df69.plot(x='Name', y='f1-score', kind='bar')

print(f'\n\nbest model: {best_model_name}, best score: {best_score}\n\n')

if best_mtype=='single':
    model = all_modelnames[best_model_name]
    pred = model.predict(X_test)
    for i in range(0,50):
        if pred[i]== 0:
            print([i],'Dementia')
        else:
            print([i],'Non Dementia')
elif best_mtype=='double':
    modelnames = best_model_name.split(', ')
    model1 = all_modelnames[modelnames[0]]
    model2 = all_modelnames[modelnames[1]]
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_train)
    model2.fit(X_train, pred1)
    pred = model2.predict(X_test)
    for i in range(0,50):
        if pred[i]== 0:
            print([i],'Dementia')
        else:
            print([i],'Non Dementia')
elif best_mtype=='stacked':    
    estimator_list = []
    for i in layer_stack:
        estimator_list.append((i.__class__.__name__, i))
    stack_model = StackingClassifier(estimators=estimator_list, 
                                     final_estimator= LogisticRegression())
    stack_model.fit(X_train, y_train)
    pred = stack_model.predict(X_test)
    for i in range(0,50):
        if pred[i]== 0:
            print([i],'Dementia')
        else:
            print([i],'Non Dementia')