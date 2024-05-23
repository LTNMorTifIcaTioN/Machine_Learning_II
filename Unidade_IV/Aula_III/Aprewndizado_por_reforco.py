# Boosting - Definição
"""
Contextualização
> Ensemble Learning:
    >> Utilização de vários modelos fracos em conjunto.
    >> Métodos tradicionais.
> Métodos tradicionais:
    >> Bagging.
    >> Boosting.

Boosting
> Agrega vários alunos fracos para a formação de um modelo forte final.
> Modelo sequencial.
> Aprendizagem através dos erros anteriores.
> Funcionamento:
    1. Geração de vários weak learners.
    2. Identificação de previsões falsas.
    3. Repetir até que saída seja satisfatória.

Boosting x Bagging
> Boosting:
    >> Mais simples.
    >> Treinamento paralelo.
    >> Combinação dos resultados (votação, média, etc).
> Boosting:
    >> Treinamento sequencial.
    >> Atribuição de pesos (votação ponderada).
"""
#%%
# Boosting – Modelo
"""
Contextualização
> Modelos:
    >> Geralmente simples.
> Modelos mais usados:
    >> Boosting Adaptativo (ADABoost).
    >> Gradient Boost.
    >> Aumento de Gradiente Extremo (XGBoost).

Reforço de Gradiente
    > Treina novos modelos conforme o erro dos modelos anteriores;
    > Machine Learning:
        >> Supervisionado: Prediz o próximo valor.
        >> Não Supervisionado: Identifica grupos.
        >> Reforço: Aprende com erros.

Gradient Boost
> Treina novos modelos conforme o erro dos modelos anteriores.
> Descida por Gradiente:
    >> X-entrada e y-Saída.
    >> Treina-se um modelo M1 sobre (x,y);
    >> Treina-se um modelo R sobre o resíduo (y-M1(x))
    >> Cria-se um novo modelo M2(x) = M1(x) + gama * R(x), onde gama é o peso dado ao modelo.
> Funcionamento:
    >> Passo 1: construção do 1º modelo;  
    >> Passo 2: construção do 2º modelo;
    >> Passo 3: construção modelo final;
Parâmetros:
    >> Número de estimadores;
    >> Taxa de aprendizagem;
    >> Critério de parada;
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
#%%
# Load the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Create a Gradient Boosting Regressor
gb_clf = GradientBoostingRegressor(loss='squared_error', n_estimators=100, learning_rate=0.1, random_state=42)
#%%
# Train the Gradient Boosting Regressor
gb_clf.fit(X_train, y_train)
#%%
# Predict the values for the test data
y_pred = gb_clf.predict(X_test)
#%%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
#%%
# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), gb_clf.feature_importances_, align='center')
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.xticks(range(X.shape[1]), housing.feature_names, rotation=90)
plt.show()
#%%
# Boosting – Aplicações
"""
Contextualização
> O Boosting é responsável por agregar vários modelos fracos.
    > Melhor desempenho.

Aplicações do Boosting
> Saúde
    >> Previsões de dados médicos.
    >> Seleção de genes ou proteínas para identificar uma característica específica de interesse.
> Tecnologia da Informação
    >> Sistemas de detecção de intrusão de rede.
> Meio Ambiente
    >> Mapeamento dos tipos de áreas em uma paisagem.
> Finanças
    >> Detecção de fraude, avaliações de risco de crédito, problemas de precificação, etc.
"""