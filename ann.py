
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('Straining_Actions.csv')
X = dataset.iloc[:, 3:5].values
y = dataset.iloc[:, 1:3].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
x_trans = MinMaxScaler(feature_range = (0, 1))
y_trans = MinMaxScaler(feature_range = (0, 1))
X_train = x_trans.fit_transform(X_train)
X_test = x_trans.transform(X_test)
y_train = y_trans.fit_transform(y_train)
y_test = y_trans.transform(y_test)

model = Sequential()
model.add(Dense(output_dim = 100, activation = 'relu', input_dim = 2))
model.add(Dense(output_dim = 100, activation = 'relu'))
model.add(Dense(output_dim = 2))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.fit(X_train, y_train, batch_size = 1, nb_epoch = 2000)
y_pred = model.predict(X_test)
y_predtrns = y_trans.inverse_transform(y_pred)
y_testtrns = y_trans.inverse_transform(y_test)