# Classificação
"""
> É o ato de receber um novo registro, o algoritmo consiga prever a que classe o dado pertence, por meio da análise dos
atributos que esse registro possui.

> Funções de Classificação:
> Função Binária (Binary Classification)
> Função Multiclasse (Multiclass Classification)
"""

# Classificação Não-Linear
"""
> Técnica utilizada quando a base de dados não possui classes linearmente separáveis.

Base de dados Iris
> Base de dados que contém 3 classes (setosa, versicolor e virginica) e 4 atributos (sepal length, sepal width, petal
length e petal width).
"""
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Carregar a base de dados Iris
iris = load_iris()

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
#%%
# Criar um modelo de Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)
#%%
# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = [i for i in range(3)]
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)

# Adicionar os nomes das classes às linhas e colunas da matriz de confusão
plt.xlabel("Previsões")
#%%
cm
#%%
# Plotar as previsões
plt.figure(figsize=(10, 6))

# Plotar as previsões corretas
plt.scatter(X_test[y_test == y_pred][:, 0], X_test[y_test == y_pred][:, 1], c='green', label='Correto')

# Plotar as previsões erradas
plt.scatter(X_test[y_test != y_pred][:, 0], X_test[y_test != y_pred][:, 1], c='red', label='Incorreto')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Comparação das Previsões com as Classes Reais")
plt.legend()
plt.show()
#%%
from sklearn.metrics import accuracy_score

# Calcular o p-valor do treino
train_score = model.score(X_train, y_train)
print("P-valor do treino:", train_score)

# Calcular o p-valor do teste
test_score = model.score(X_test, y_test)
print("P-valor do teste:", test_score)

# Calcular o p-valor das previsões
prediction_score = accuracy_score(y_test, y_pred)
print("P-valor das previsões:", prediction_score)
#%%
# Aplicação regressão logística: classificação
"""
Portas Lógicas:

A	B	    AND	    OR	    NAND	NOR	    XOR
0	0	    0	    0	    1	    1	    0
0	1	    0	    1	    1	    0	    1
1	0	    0	    1	    1	    0	    1
1	1	    1	    1	    0	    0	    0

AND: Retorna 1 se ambas as entradas forem 1.
OR: Retorna 1 se pelo menos uma das entradas for 1.
NAND: Retorna o oposto de AND, ou seja, 1 se pelo menos uma das entradas for 0.
NOR: Retorna o oposto de OR, ou seja, 1 se ambas as entradas forem 0.
XOR (ou exclusivo): Retorna 1 se as entradas forem diferentes.
"""
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Carregar a base de dados Iris
iris = load_iris()

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Criar um modelo de Regressão Logística
model = LogisticRegression(max_iter=200)

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcular o p-valor do treino
train_score = model.score(X_train, y_train)
print("P-valor do treino:", train_score)

# Calcular o p-valor do teste
test_score = model.score(X_test, y_test)
print("P-valor do teste:", test_score)

# Calcular o p-valor das previsões
prediction_score = accuracy_score(y_test, y_pred)
print("P-valor das previsões:", prediction_score)

# Plotar os resultados do treino
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Treino")

# Plotar os resultados do teste
plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Teste")

# Plotar os resultados das previsões
plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set1)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Previsões")

plt.tight_layout()
plt.show()
#%%
# Curva ROC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Binarize as saídas
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

# Calcular as probabilidades previstas para as classes
y_probs = model.predict_proba(X_test)

# Calcular a curva ROC para cada classe
plt.figure(figsize=(8, 6))
for i in range(3):  # 3 classes no dataset Iris
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Classe %d (AUC = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC da Regressão Logística para múltiplas classes')
plt.legend()
plt.show()
#%%
