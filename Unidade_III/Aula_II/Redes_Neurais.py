# Redes neurais – Definições
"""
Contextualização:
> As Redes Neurais Artificiais (RNAs) são modelos matemáticos inspirados no cérebro humano. Elas aprendem (ou são
treinadas) processando exemplos, cada um dos quais contém uma entrada e um resultado conhecido, formando associações
ponderadas entre os dois, armazenadas na estrutura de dados da própria rede.

Definição de Redes Neurais Artificiais
> Uma RNA é baseada em um conjunto de nós conectados ou neurônios artificiais, que modelam os neurônios em um cérebro
biológico.

> Existe uma classe de redes, chamada SOM (Self-Organizing Map), que se enquadram em aprendizado não supervisionado
que seus resultados são baseados em similaridades das entradas. As RNAs tem diversas aplicações, das mais básicas as
mais avançadas.

> Uma RNA é baseada em um conjunto de nós conectados, ou neurônios artificiais, que modelam os neurônios em cérebro
biológico. Em cada conexão, como as sinapses em um cérebro, pode transmitir um sinal a outros neuronios.
"""
#%%
# Redes neurais - Modelo matemático
"""
Contextualização:
> Os primeiros modelos matemáticos das RNAs foram propostos por MCCULLOCH E PITTS em 1943.

Redes Neurais - modelo matemático:
> As RNAs são compostas por neurônios artificiais que são conceitualmente derivados de neurônios biológicos.
Cada neurônio artificial tem entradas e produz uma única saída que pode ser enviada a vários outros neurônios.

Passos para a modelagem de um neurônio simples:
> Passo 1: Camada de entrada de uma RNA;
> Passo 2: Pesos de uma RNA;
> Passo 3: Somatória das entradas com os pesos;
    u = w1x1 + w2x2 + ... + wnxn + Teta, onde:
        w: pesos
        x: entradas
        Teta = limiar/bias (entradas constantes)
        u: ativação.
> Passo 4: Função de ativação;
    y(t) = + 1 se u > 0 ou -1 se u < 0.
"""
#%%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
#%%
# Função de ativação
def activation_function(u):
    return 1 if u > 0 else -1
#%%
# Passo 1: Camada de entrada de uma RNA
iris = datasets.load_iris()
X = iris.data
y = iris.target
#%%
# Passo 2: Pesos de uma RNA
num_features = X.shape[1]
weights = np.random.randn(num_features)
#%%
# Passo 3: Somatória das entradas com os pesos
bias = 0
predictions = []
for i in range(len(X)):
    u = np.dot(X[i], weights) + bias
    activation = activation_function(u)
    predictions.append(activation)
#%%
# Passo 4: Função de ativação
predictions = np.array(predictions)
#%%
# Plotar o resultado
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.xlabel('Comprimento da Pétala')
plt.ylabel('Largura da Pétala')
plt.title('RNA Neurônio Simples com Função de Ativação')
plt.colorbar()
plt.show()
#%%
# Imprimir o valor do p-valor do treino
p_value_train = np.sum(predictions == y) / len(y)
print("P-valor do treino:", p_value_train)
#%%
# Redes neurais – Aplicações
"""
Contextualização:
> As RNAs possuem diversas aplicações sendo executadas até os dias atuais, como forma de pesquisa também nas indústrias.
Hoje as redes neurais estão ajudando os humanos a sobreviverem às transições da nova era nos setores educacional,
financeiro, aeroespacial e automotivo.

Aplicações:
> Como esses neurônios artificiais funcionam de forma semelhante ao cérebro humano, eles podem ser usados para 
reconhecimento de imagem, reconhecimento de personagem e previsões do mercado de ações.
> Trabalho de Jian, Z.H.U. (2010), aplicações das RNAs com análise de sentimentos.
"""