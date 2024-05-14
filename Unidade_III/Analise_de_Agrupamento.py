# Análise de agrupamento – Conceito e tipos
"""
Contextualização:
> Agrupamentos consistem em encontrar grupos entre os objetos;
> Tipo de aprendizado não supervisionado;
> Os objetos de um grupo devem ser mais similares ou relacionados entre si do que a objetos de outro grupo;

Análise de agrupamento:
> Dado um conjunto de dados não classificados, descobrir as classes dos elementos (grupos ou clusters) e possivelmente
o número de grupos existentes a partir de suas características.

Distância inter-cluster (IC): Entre objetos de grupos diferentes
Distância intra-cluster (IC): Entre objetos de um mesmo grupo

> Cluster é um conceito subjetivo;
> Cluster é um conceito ambíguo;

Tipos de agrupamento:
> Hierárquico;
> Não Hierárquico;

"""
#%%
# k-médias – Definição e aplicação
"""
Contextualização:
> O K-Médias foi proposto por J Mac Queen em 1967, e é um dos mais conhecidos e utilizados, além de ser o que possui o
maior número de variações;
> É um algorítmo de agrupamento de objetos baseados em atributos em um número K de grupos;

K-Médias:
> Dado um conjunto de dados não classificados, descobrir as classes dos elementos (grupos ou clusters) e possivelmente
o número de grupos existentes a partir de suas características.

Algoritmo do K-médias:
> Passo 1: Selecione o número K para decidir o número de clusters;
> Passo 2: Selecione K pontos aleatórios ou centróides;
> Passo 3: Atribua cada ponto de dados ao seu centroide mais próximo, que formará os K clusters predefinidos;
> Passo 4: Calcule a variância e coloque um novo centróide de cada cluster;
> Passo 5: Repita os passos 3 e 4, para achar o cluster mais próximo de cada centróide;
> Passo 6: Se ocorrer alguma reatribuição, vá para o passo 4, se não, para o final;
> Passo 7: O modelo está pronto.
"""
#%%
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
#%%
# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data
#%%
# Passo 1: Selecione o número K para o número de clusters
k = 3
#%%
# Passo 2: Selecione K centróides aleatórios
centroids = X[np.random.choice(range(X.shape[0]), k, replace=False)]
#%%
# Passo 3: Atribua cada ponto de dados ao seu centróide mais próximo, formando K clusters predefinidos
clusters = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    distances = np.linalg.norm(X[i] - centroids, axis=1)
    clusters[i] = np.argmin(distances)
#%%
# Passo 4: Calcule a variância e coloque um novo centróide para cada cluster
new_centroids = np.zeros((k, X.shape[1]))
for i in range(k):
    points = X[clusters == i]
    if points.shape[0] > 0:
        new_centroids[i] = np.mean(points, axis=0)
#%%
# Passo 5: Repita os passos 3 e 4 até que não haja mais reatribuição
while True:
    old_centroids = centroids.copy()
    centroids = new_centroids
    clusters = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        clusters[i] = np.argmin(distances)
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[clusters == i]
        if points.shape[0] > 0:
            new_centroids[i] = np.mean(points, axis=0)
    if np.allclose(old_centroids, centroids):
        break
#%%
# Plotando os clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=2)
plt.xlabel('Comprimento da Pétala')
plt.ylabel('Largura da Pétala')
plt.title('Agrupamento pelo Algoritmo K-Means do Conjunto de Dados Iris')
plt.show()
#%%
# Avaliação do modelo
silhouette_avg = silhouette_score(X, clusters)
#%%
# Imprimir o valor do silhouette score
print("Silhouette Score:", silhouette_avg)
#%%
# Agrupamento espectral – Definição e aplicação
"""
Contextualização:
> O Agrupamento Espectral usa a abordagem de conectividade para agrupamento, em que comunidades de nós (ou seja, pontos
de dados) que estão conectados ou imediatamente próximos uns dos outros são identificados em um gráfico;
> O agrupamento espectral usa informações dos valores próprios (espectro) de matrizes especiais (ou seja, matriz de
afinidade, matriz de graus e matriz laplaciana) derivadas do gráfico ou conjunto de dados.
> Não faz suposições sobre a forma dos clusters. As técnicas de agrupamento, como K-Means, assumem que os pontos
atribuídos a um agrupamento são esféricos em torno do centro do agrupamento. Esta é uma suposição forte e pode nem
sempre ser relevante. Nesses casos, o Agrupamento espectral ajuda a criar clusters mais precisos.

Algoritmo de Agrupamento Espectral
> Passo 1: Forme uma matriz de distância;
> Passo 2: Transforme a matriz de distância em uma matriz de afinidade A;
> Passo 3: Calcule a matriz de graus D e a matriz laplaciana L = D-A;
> Passo 4: Encontre os autovalores e autovetores de L;
> Passo 5: Com os autovetores de k maiores autovalores calculados na etapa anterior, forme uma matriz;
> Passo 6: Normalize os vetores;
> Passo 7: Agrupe os pontos de dados no espaço k-dimensional;
"""
#%%
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
#%%
# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
#%%
# Passo 1: Forme uma matriz de distância
# Neste exemplo, usaremos a matriz de distância euclidiana
distance_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
#%%
# Passo 2: Transforme a matriz de distância em uma matriz de afinidade A
affinity_matrix = np.exp(-distance_matrix ** 2)
#%%
# Passo 3: Calcule a matriz de graus D e a matriz laplaciana L = D-A
degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
laplacian_matrix = degree_matrix - affinity_matrix
#%%
# Passo 4: Encontre os autovalores e autovetores de L
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
#%%
# Passo 5: Com os autovetores de k maiores autovalores calculados na etapa anterior, forme uma matriz
k = 3  # Número de clusters desejado
top_k_eigenvectors = eigenvectors[:, 1:k+1]
#%%
# Passo 6: Normalize os vetores
normalized_vectors = top_k_eigenvectors / np.linalg.norm(top_k_eigenvectors, axis=1)[:, np.newaxis]
#%%
# Passo 7: Agrupe os pontos de dados no espaço k-dimensional
model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
clusters = model.fit_predict(normalized_vectors)
#%%
# Avaliação do modelo
silhouette_avg = silhouette_score(X, clusters)
#%%
# Plotando o resultado
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Comprimento da Pétala')
plt.ylabel('Largura da Pétala')
plt.title('Agrupamento Espectral do Conjunto de Dados Iris')
plt.colorbar()
plt.show()
#%%
# Imprimir o valor do silhouette score
print("Silhouette Score:", silhouette_avg)
#%%
"""
# Silhouette Score:
O Silhouette Score é uma métrica de avaliação utilizada para avaliar a qualidade de agrupamento em um conjunto de dados.
Ele mede a similaridade entre cada ponto de dados e o cluster ao qual ele pertence, comparando-o com a similaridade
com os outros pontos de dados no mesmo cluster.

O Silhouette Score é calculado da seguinte forma:
    1. Para cada ponto de dados, é calculada a média das distâncias entre esse ponto e todos os outros pontos do mesmo
    cluster.
    2. Também é calculada a média das distâncias entre esse ponto e todos os pontos dos outros clusters.
    3. O Silhouette Score para o ponto de dados é a diferença entre essas duas médias, dividida pela distância mínima
    entre o ponto e qualquer ponto do mesmo cluster.
    4. O Silhouette Score para todos os pontos de dados é a média desses valores.

> Um Silhouette Score positivo indica que o ponto de dados está bem agrupado, enquanto um Silhouette Score negativo
indica que o ponto de dados está mal agrupado.
> Um Silhouette Score próximo de zero indica que o ponto de dados está entre clusters com pouca similaridade.
"""