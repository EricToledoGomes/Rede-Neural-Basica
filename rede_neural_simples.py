import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Definir o conjunto de dados
X = np.array([[1.5], [1.6], [1.7], [1.8], [1.9]])
Y = np.array([[60], [65], [70], [75], [80]])

# Definir o modelo de rede neural
model = Sequential()
model.add(Dense(1, input_dim=1))

# Compilar o modelo
model.compile(loss='mean_squared_error', optimizer='sgd')

# Treinar o modelo
model.fit(X, Y, epochs=1000, verbose=0)

# Fazer uma previsão
x_test = np.array([[1.75]])
y_pred = model.predict(x_test)

# Exibir o resultado
print("A pessoa com altura", x_test[0][0], "m deve pesar cerca de", y_pred[0][0], "kg")