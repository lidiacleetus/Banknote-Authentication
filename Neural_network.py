import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/drive/MyDrive/BankNote_Authentication.csv')
print(data.head())

#dimensions of data
print(data.shape)

#statistical description
print(data.describe())

#Missing values
data.isnull().sum()  #There were no missing values in the data

#Datatype of all variables
data.dtypes

#Plot Histogram 
data.hist()
plt.show()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(data[['variance', 'skewness', 'curtosis', 'entropy']])
scaled_features = pd.DataFrame(scaled_features)
scaled_features.head()

#Separate feature and target variables
X = scaled_features.values[:,]
Y = data.values[:,-1]

#Train and test
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.3)


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Gradient Descent Function
def grad_descent(X,y,learning_rate=0.01,iterations=2000):
  m = X.shape[0]
  ones = np.ones((m,1))
  X = np.concatenate((ones,X),axis=1)
  n = X.shape[1]
  theta = np.zeros(n)
  h = np.dot(X,theta)

  #gradient descent alg
  cost = np.zeros(iterations)
  for i in range(iterations):
    theta[0] = theta[0] - (learning_rate/m)*sum(h-y)
    for j in range(1,n):
      theta[j] = theta[j] - (learning_rate/m)*sum((h-y)*X[:,j])
    h = np.dot(X,theta)
    cost[i] = 1/(2*m) * sum(np.square(h-y))
  return cost,theta
  
cost, theta = grad_descent(X,Y)
print(theta)
print(cost[-1])

print('Theta0:   %0.3f' %theta[0],'\nTheta1:   %0.3f' %theta[1],'\nTheta2:   %0.3f' %theta[2],
      '\nTheta3:   %0.3f' %theta[3],'\nTheta4:   %0.3f' %theta[4])
print("\nFinal cost:    %0.3f" %cost[-1])


#Custom activation function
z = theta[0] + theta[1]*X.transpose()[0] + theta[2]*X.transpose()[1] + theta[3]*X.transpose()[2] + theta[4]*X.transpose()[3]

from keras import backend as K
def custom_activation(z):
  return K.sigmoid(z)
  
  
#Model
n_feature = X.shape[1]
inputShape = (n_feature,)
model = Sequential()
model.add(Dense(1372,activation='relu',input_shape=inputShape))
model.add(Dense(1,activation = custom_activation))

from keras.optimizers import SGD
opt = SGD(learning_rate=10E-3,momentum=0.9)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

history=model.fit(X_train,y_train,
                  epochs=25,
                  batch_size=32,
                  validation_data=(X_test,y_test))


#Plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,101)

#Train vs test loss
plt.title('Loss')
plt.plot(loss,label='train')
plt.plot(val_loss, label='test')
plt.legend()

plt.figure()
#Train vs test accuracy
plt.title('Accuracy')
plt.plot(acc,label='train')
plt.plot(val_acc, label='test')
plt.legend()
plt.show()

#Loss function vs epochs
fig,ax = plt.subplots(figsize = (10,6))
ax.set_xlabel('Iterations')
ax.set_ylabel('J(Theta)')
_=ax.plot(range(2000),cost,'b.')


#F1-score
y_pred = model.predict(X_test,verbose=0)
yhat_class = model.predict_classes(X_test,verbose=0)
y_pred = y_pred[:,0]
yhat_class = yhat_class[:,0]

from sklearn.metrics import f1_score
F1Score = f1_score(y_test,yhat_class)
print(F1Score)
