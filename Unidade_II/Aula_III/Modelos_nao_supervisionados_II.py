# Modelo Não Supervisionado - PCA
"""
O Modelo Não Supervisionado PCA (Análise de Componentes Principais) é uma técnica de aprendizado de máquina usada
para reduzir a dimensionalidade de um conjunto de dados, enquanto tenta manter o máximo de informações possível.

Este modelo é particularmente útil quando os dados têm muitas variáveis e não podem ser facilmente visualizados.
A redução de dimensionalidade ajuda a encontrar as variáveis mais importantes nos dados e a reduzir o ruído.

O PCA é baseado na variância dos dados, ou seja, ele tenta criar uma nova representação dos dados,
com uma dimensão menor, mantendo a variância entre eles.

Existem diferentes tipos de agrupamento que podem ser realizados com o PCA, incluindo agrupamento exclusivo, sobreposto,
hierárquico e probabilístico.

Por exemplo, o agrupamento exclusivo estipula que um ponto de dados pode existir apenas em um cluster.
O algoritmo de clusterização k-médias é um exemplo de agrupamento exclusivo.

Em contraste, o agrupamento sobreposto permite que pontos de dados pertençam a vários clusters com diferentes níveis
de filiação.
"""
#%%
# Modelo de Decomposição em Valores Singulares (SVD)
"""
O Modelo de Decomposição em Valores Singulares (SDV), também conhecido como Singular Value Decomposition (SVD),
é um método muito útil para a análise de sistemas multivariáveis.

A SVD é baseada na ideia de que qualquer matriz pode ser decomposta em uma combinação linear de matrizes mais simples.
Essas matrizes mais simples são chamadas de matrizes singulares e representam as características fundamentais da matriz original.

A SVD decompõe a matriz num produto dos fatores de outras três matrizes:
> A=USV, onde U e V são matrizes ortogonais (MMT=I) e S é diagonal.

Os valores da matriz diagonal são chamados de valores singulares e por isso a decomposição recebe este nome.

O número de valores singulares diferentes de zero é igual ao rank (posto) da matriz.

A decomposição em valores singulares vale tanto para matrizes quadradas quanto retangulares.
A matriz pode ter elementos reais ou complexos.
A SVD é usada em álgebra linear para minimizar erros computacionais em operações com matrizes de grande porte.

A SVD é implementada na linguagem Wolfram como SingularValueDecomposition[m],
que retorna uma lista {U, D, V},
onde U e V são matrizes e D é uma matriz diagonal composta pelos valores singulares de m.
"""
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Carregar a base de dados Iris
iris = load_iris()

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Criar um modelo de SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)  # Reduzir para 2 componentes principais
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

# Criar um modelo de Regressão Logística
model = LogisticRegression(max_iter=200)

# Treinar o modelo com os dados de treinamento
model.fit(X_train_svd, y_train)

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test_svd)

# Calcular o p-valor do treino
train_score = model.score(X_train_svd, y_train)
print("P-valor do treino:", train_score)

# Calcular o p-valor do teste
test_score = model.score(X_test_svd, y_test)
print("P-valor do teste:", test_score)

# Calcular o p-valor das previsões
prediction_score = accuracy_score(y_test, y_pred)
print("P-valor das previsões:", prediction_score)

# Plotar os resultados do treino
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(X_train_svd[:, 0], X_train_svd[:, 1], c=y_train, cmap=plt.cm.Set1)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Treino")

# Plotar os resultados do teste
plt.subplot(1, 3, 2)
plt.scatter(X_test_svd[:, 0], X_test_svd[:, 1], c=y_test, cmap=plt.cm.Set1)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Teste")

# Plotar os resultados das previsões
plt.subplot(1, 3, 3)
plt.scatter(X_test_svd[:, 0], X_test_svd[:, 1], c=y_pred, cmap=plt.cm.Set1)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Previsões")

plt.tight_layout()
plt.show()
#%%
# Modelo de Alocação Latente de Dirichlet (LDA)
"""
O Modelo de Alocação Latente de Dirichlet (LDA), é um modelo estatístico utilizado para descobrir tópicos ocultos em um
conjunto de documentos1. Este modelo é amplamente utilizado em áreas como processamento de linguagem natural,
mineração de texto e análise de dados.

A LDA se baseia na ideia de que cada documento é uma mistura de um conjunto de tópicos e que cada tópico é uma mistura
de um conjunto de palavras2. Em outras palavras, o LDA acredita que cada documento é composto por vários tópicos e que
a presença de cada palavra é atribuível a um dos tópicos do documento.

A LDA é implementada através de um processo de inferência estatística que permite identificar os tópicos subjacentes em
um corpus de texto e as palavras mais relevantes em cada tópico.

Os principais parâmetros para o modelo são:

> α que indica de quantos tópicos os documentos são compostos. Quanto maior, maior a quantidade de tópicos, mais
específica a distribuição.

> β que indica de quantas palavras os tópicos são compostos. Quanto maior, maior a quantidade de palavras, mais
específica a distribuição.

Os principais outputs que se extraem do modelo, além dos tópicos em si, são a frequência de palavras por tópicos
e a distribuição dos tópicos para cada documento. Ou seja, quanto percentualmente aquele tópico é relevante para
documento.
"""
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Carregar a base de dados Iris
iris = load_iris()

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# Aplicar a transformação logarítmica para tornar os valores não negativos
X_non_negative = np.log1p(X_scaled - X_scaled.min())

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_non_negative, iris.target, test_size=0.2, random_state=42)

# Criar um modelo de LDA
lda = LatentDirichletAllocation(n_components=2)  # Reduzir para 2 componentes latentes
X_train_lda = lda.fit_transform(X_train)
X_test_lda = lda.transform(X_test)

# Criar um modelo de Regressão Logística
model = LogisticRegression(max_iter=200)

# Treinar o modelo com os dados de treinamento
model.fit(X_train_lda, y_train)

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test_lda)

# Calcular o p-valor do treino
train_score = model.score(X_train_lda, y_train)
print("P-valor do treino:", train_score)

# Calcular o p-valor do teste
test_score = model.score(X_test_lda, y_test)
print("P-valor do teste:", test_score)

# Calcular o p-valor das previsões
prediction_score = accuracy_score(y_test, y_pred)
print("P-valor das previsões:", prediction_score)

# Plotar os resultados do treino
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap=plt.cm.Set1)
plt.xlabel("Componente Latente 1")
plt.ylabel("Componente Latente 2")
plt.title("Treino")

# Plotar os resultados do teste
plt.subplot(1, 3, 2)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap=plt.cm.Set1)
plt.xlabel("Componente Latente 1")
plt.ylabel("Componente Latente 2")
plt.title("Teste")

# Plotar os resultados das previsões
plt.subplot(1, 3, 3)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_pred, cmap=plt.cm.Set1)
plt.xlabel("Componente Latente 1")
plt.ylabel("Componente Latente 2")
plt.title("Previsões")

plt.tight_layout()
plt.show()
#%%
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
#%%
data = load_iris()
#%%
X = data.data
#%%
pca = PCA(n_components=2)
pca.fit(X)
#%%
print(pca.explained_variance_ratio_)
#%%
import matplotlib.pyplot as plt

# Variação explicada do PCA
explained_variance_ratio = pca.explained_variance_ratio_

# Criar o gráfico de barras
plt.figure(figsize=(8, 6))
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio)
plt.xlabel('Componente Principal')
plt.ylabel('Variação Explicada (%)')
plt.title('Variação Explicada pelo PCA')
plt.ylim(0, 1)
plt.show()