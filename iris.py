import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

classe_dummy = np_utils.to_categorical(classe)
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units= 4, activation='relu', input_dim=4))
classificador.add(Dense(units= 4, activation='relu'))
classificador.add(Dense(units= 3, activation='softmax'))
classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10,
                  epochs=1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.9)
print(f'{previsoes}')

classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(previsoes2, classe_teste2)
print(matrix)
