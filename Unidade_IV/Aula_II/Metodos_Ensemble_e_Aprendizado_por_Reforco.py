# Bagging: definição
"""
Contextualização
> Ensemble Learning:
    >> Vários weak learners agregados formam um strong learner.
    >> Pode reduzir a variância e o viés.
    >> Melhor desempenho do modelo.

> Tipos de Ensemble tradicionais:
    >> Bagging.
    >> Boosting.

Bootstrapping
> Gera amostras de tamanho B do conjunto de dados inicial de tamanho N de maneira aleatória com reposição.

Bagging
> Gera um conjunto de dados por amostragem bootstrap dos dados originais.
> Classificação:
    >> Resultado de cada estimador é uma base.
    >> A mais votada, é definida como resultado final.
> Regressão:
    >> Média dos resultados de cada um dos estimadores.
> Vantagens:
    >> Redução da variância e viés.
    >> Estabilidade e robustez.
    >> Melhor desempenho do modelo.
> Desvantagens
    >> Alto custo computacional.
    >> Funcionamento correto apenas se o modelo base possui bom desempenho.
"""
#%%
# Bagging – Modelo
"""
Contextualização
> Ensemble Learning:
    >> Bagging.
    >> Boosting.

Modelos
> Encontrar o tradeoff (equilíbrio) entre variância e viés

Bagging
> Combinação dos weak learners.
> Escolher um modelo base.
> Modelos homogêneos.
> Treinamento dos weak learners.
    >> Paralelo e independente.
> Combinação dos resultados.
    >> Medidas ou técnicas estatísticas.
> Exemplo de construção de modelo:
    >> Escolher um modelo base:
        >>> Arvore de decisão
        >>> KNN
        >>> RNA
"""
#%%
# Bagging – Aplicação
"""
Contextualização
> O Bagging é responsável por agregar vários modelos fracos.
> Tem melhor desempenho.

Aplicações do Bagging
> Saúde:
    >> Previsões de dados médicos.
    >> Seleção de genes ou proteínas para identificar uma característica específica de interesse.
> Tecnologia da Informação
    >> Sistemas de intrusão de rede.
> Meio Ambiente:
    >> Mapeamento dos tipos de áreas em uma paisagem.
> Finanças:
    >> Detecção de fraude, avaliações de risco de crédito, problemas de precificação, etc.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#%%
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Create a Decision Tree classifier as the base model
base_model = DecisionTreeClassifier(random_state=42)
#%%
# Create a Bagging classifier with the base model
bagging_clf = BaggingClassifier(base_model, n_estimators=10, max_samples=0.5, max_features=0.5, random_state=42)
#%%
# Train the Bagging classifier
bagging_clf.fit(X_train, y_train)
#%%
# Predict the labels for the test data
y_pred = bagging_clf.predict(X_test)
#%%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#%%
# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(3), iris.target_names)
plt.yticks(np.arange(3), iris.target_names)
plt.colorbar()
plt.show()
#%%
