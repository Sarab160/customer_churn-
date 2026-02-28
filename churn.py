import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("customer.csv")
# print(df.head(3))


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Optional: first convert to numeric
df['TotalCharges'].fillna(df["TotalCharges"].mean(), inplace=True)

# print(df.isnull().sum())
# print(df.duplicated().sum())


# sns.pairplot(data=df,hue="Churn")
# plt.title("Nature of Dataset")
# plt.show()
# plt.savefig("Nature.png")
# print(df.info())
X=df[["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]]
Y=df["Churn"]

feature=df[["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encode_array=ohe.fit_transform(feature)
get_columns=ohe.get_feature_names_out(feature.columns)
encode_data=pd.DataFrame(encode_array,columns=get_columns)
Y=ohe.fit_transform(df[["Churn"]])

print(df.isnull().sum())
x_final =pd.concat([X,encode_data],axis=1)

ru=RandomOverSampler()
x_ru,y_ru=ru.fit_resample(x_final,Y)

x_train,x_test,y_train,y_test=train_test_split(x_ru,y_ru,test_size=0.2,random_state=42)

dtc=RandomForestClassifier(n_estimators=100,n_jobs=100)

dtc.fit(x_train,y_train)

print("Accuracy",dtc.score(x_test,y_test))

cf=confusion_matrix(y_test,dtc.predict(x_test))
print("Precision Score",precision_score(y_test,dtc.predict(x_test)))
print("Recall Score",recall_score(y_test,dtc.predict(x_test)))
print("F1 Score",f1_score(y_test,dtc.predict(x_test)))
print(classification_report(y_test, dtc.predict(x_test)))
sns.heatmap(cf,annot=True)
plt.show()

# for i in  range(1,30):
#     dtc2=DecisionTreeClassifier(max_depth=i)
#     dtc2.fit(x_train,y_train)
#     print(i,dtc2.score(x_train,y_train),dtc2.score(x_test,y_test))