# importing the libraries
import numpy   as np
import pandas  as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss










# setting options for libraries




pd.set_option('display.max_columns', None)

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print(f'train shape {train.shape}')
print(f'test shape {test.shape}')

train.columns

train.isna().sum() ,test.isna().sum()

data = []
for index,row in train.iterrows():
  for char in range(len(row['plain_text'])):
      data.append([row['ID']+f'_{char}', row['plain_text'][char], row['encrypted_text'][char], row['encryption_key'][0],row['encryption_key'][1],row['encryption_key'][2]])

trainn = pd.DataFrame(data=data,columns=['ID', 'plain_text', 'encrypted_text', 'encryption_key0','encryption_key1','encryption_key2'])

trainn = pd.get_dummies(trainn.drop('ID',axis=1))


TARGET  = [x for x in trainn.columns if 'plain' in x]
COLUMNS = [x for x in trainn.columns if x not in TARGET]

X = trainn[COLUMNS].copy()
y = trainn[TARGET].copy()

# creating the trainnig, cross validation, test data sets (80/20/20)










X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val , y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

model = keras.Sequential([

    layers.Dense(units=26, activation='relu'),
    layers.Dense(units=26, activation='relu'),
    layers.Dense(units=26, activation='relu'),
    layers.Dense(units=26, activation='relu'),

    layers.Dense(units=26, activation='softmax'),

])

model.compile(optimizer='Adam', loss=tf.compat.v1.losses.log_loss)
model.fit(x=X_train, y=y_train, epochs=3)



yhat_train  = model.predict(X_train)
yhat_val    = model.predict(X_val)
yhat_test   = model.predict(X_test)

print(f'trinnig error  : {log_loss(y_train,yhat_train)}')
print(f'val error      : {log_loss(y_val,yhat_val)}')
print(f'test error     : {log_loss(y_test,yhat_test)}')


model.fit(x=X,y=y,epochs=3)
data = []
for index,row in test.iterrows():
  for char in range(len(row['encrypted_text'])):
      data.append([row['ID']+f'_{char}', row['encrypted_text'][char], row['encryption_key'][0],row['encryption_key'][1],row['encryption_key'][2]])

testt = pd.DataFrame(data=data,columns=['ID', 'encrypted_text', 'encryption_key0', 'encryption_key1', 'encryption_key2'])
ids   = testt['ID'].copy()
testt = pd.get_dummies(testt.drop('ID',axis=1))
yhat  = model.predict(testt)
out   = pd.concat([ids,pd.DataFrame(yhat)],axis=1)
out.to_csv('submit.csv',index=0)