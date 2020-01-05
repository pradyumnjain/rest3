import pandas as pandas
from sklearn.linear_model import LogisticRegression

#create df
train = pd.read_csv('titanic.csv')

#drop null
train.dropna(inplace=True)

#fetures and target
target = 'Survived'
features = ['pclass','Age','SibSp','Fare']

#x matrix , y vector

x = train[features]
y = train[target]

#model
model = LogisticRegression()
model.fit(x,y)
model.score(x,y)

#save model
import pickle
pickle.dump(model, open(‘model.pkl’, ‘wb’))
