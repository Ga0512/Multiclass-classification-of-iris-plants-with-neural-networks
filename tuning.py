import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)




def criarRede(optimizer, loss, kernel_initializer, activation, neurons):

    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer, input_dim = 4))


    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer))


    classificador.add(Dense(units = 3, activation = 'softmax'))

    classificador.compile(optimizer = optimizer, loss = loss,
                      metrics = ['accuracy'])

    return classificador



classificador = KerasClassifier(build_fn= criarRede)
parametro = {'batch_size': [10],
             'epochs': [1000],
             'optimizer': ['adam', 'sgd'],
             'loss': ['sparse_categorical_crossentropy'],
             'kernel_initializer': ['random_uniform', 'normal'],
             'activation': ['relu', 'softmax'],
             'neurons': [4]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametro, scoring= 'accuracy', cv=10)
grid_search = grid_search.fit(previsores, classe)

melhores_param = grid_search.best_params_
melhor_preci = grid_search.best_score_
print(melhores_param, '\n', melhor_preci)
