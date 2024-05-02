# Modelos de Regressão
"""
Elastic Net
> Para Geron (2019), a regularização é habitualmente alcançada quando restringimos os pesos do modelo. Isso ocorre
quando aplicamos Elastic Net.
> Nos modelos de regressão linear temos a possibilidade de abordar o problema da regressão de lasso e ignorar que
o problema da multicolinearidade dos regressores não existe. Para isso, existe a metodologia de regressão Elastic Net.
> Nesses casos, vai haver a ponderação entre o l1 e o l2.
    L1: Individualidade; L2: Penalidade;

> Elastic Net é um meio-termo entre a Regressão de Ridge e a Regressão Lasso.
> O termo de regularização é uma simples mistura dos termos de regularização Ridge e Lasso, e você pode controlar a taxa
de mistura r.
 R=0: Elastic Net = Ridge
 R=1: Elastic Net = Lasso
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
import numpy as np
#%%
np.random.seed(42)

# Criar dados aleatórios para as características da casa
num_features = 5
X = pd.DataFrame(np.random.rand(1000, num_features), columns=['feature_%d' % i for i in range(num_features)])

# Criar dados aleatórios para o preço da casa
y = pd.Series(np.random.randint(100000, 1000000, size=1000))

# Criar uma relação entre as características da casa e o preço da casa
beta = np.random.rand(num_features)
X['price'] = X.dot(beta) + np.random.randn(1000) * 10000

# Dividir os dados em um conjunto de treinamento e um conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X.drop('price', axis=1), X['price'], test_size=0.2, random_state=42)
#%%
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
#%%
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('Erro médio absoluto:', mae)
#%%
import matplotlib.pyplot as plt

# Plotar o modelo Elastic Net
y_pred = model.predict(X_train)
plt.plot(X_train, y_pred, color='red', label='Modelo Elastic Net')

# Adicionar legendas e rótulos aos gráficos
plt.legend()
plt.xlabel('Características da Casa')
plt.ylabel('Preço da casa')

# Mostrar o gráfico
plt.show()
#%%
# Classificação
"""
> Em CLASSIFICAÇÃO, queremos, geralmente, a previsão de classes, ao passo que na regressão, queremos prever os valores.

Matriz de Confusão
> Para calcular a matriz de confusão, primeiro precisa se ter um conjunto de previsões reais para serem comparadas
com alvos reais. Utilize a função cross_val_predict() do sklearn.model_selection.

> Assim como a função cross_val_score(), a função cross_val_predict() desempenha a validação cruzada K-fold, mas, em
vez de retornar as pontuações da avaliação, ela retorna as previsões feitas em cada parte do teste. Isso significa que
se obtém uma previsão limpa para cada instância no conjunto de treinamento.

Como Interpretar a Matriz de Confusão:
                |   Valor Predito  |
                --------------------        
                |   Sim   |   Não  |
------------------------------------
# real  |   Sim |   VP    |   FN   |
# real  |   Não |   FP    |   VN   |
------------------------------------
# """
#%%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
#%%
y_pred = cross_val_predict(model, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_pred)
print(cm)
#%%
# Ajustes no Modelo
"""
> Ajustar hiperparâmetros usando GridSearchCV para automatizar o máximo de modelos.
"""
#%%
plt.matshow(cm, cmap='gray', alpha=0.3)
plt.show()
#%%
from sklearn.model_selection import GridSearchCV
#%%
param_grid = {'alpha': [0.1, 0.5, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
#%%
print(grid_search.best_params_)
