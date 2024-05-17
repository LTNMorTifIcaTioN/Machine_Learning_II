# Sistemas de recomendação – Definição, filtros colaborativos e seleção de modelos
"""
Contextualização
> Os sistemas de recomendações são algoritmos que visam sugerir itens relevantes aos usuários.
> São usados em várias áreas, por exemplo, em serviços de vídeos, músicas, itens de lojas, dentre outros.

Definições
> Os sistemas de recomendação são sistemas projetados para recomendar itens ao usuário com base nos dados fornecidos.

> O objetivo de um sistema de recomendação é sugerir itens relevantes aos usuários. Para realizar esta tarefa,
existem duas categorias principais de métodos: métodos de filtragem colaborativa e métodos baseados em conteúdo.

1. Métodos de filtragem colaborativa: Métodos colaborativos para sistemas de recomendação são métodos que se baseiam
exclusivamente nas interações passadas registradas entre usuários e itens para produzir novas recomendações.
    >> Baseado em Modelo: Abordagens baseadas em modelo assumem um modelo 'generativo' subjacente que explica as
    interações usuário-item e tenta descobri-lo para fazer novas previsões.
    >> Baseado em Memória: Abordagens baseadas em memória trabalham diretamente com valores de interações gravadas,
    assumindo nenhum modelo, e são essencialmente baseadas na busca de vizinhos mais próximos.

2. Métodos baseados em conteúdo: O método ou filtragem baseado em conteúdo faz recomendações com base nas preferências
do usuário para os recursos do produto. Baseia-se em semelhanças entre as características dos itens.
"""
#%%
# Detecção de fraudes
"""
Contextualização
> Sistemas de detecção de fraude estão sendo mais requisitados a todo momento, pois as fraudes já se tornaram práticas
constantes pelos criminosos.

Detecção de fraude
> A abordagem básica para detecção de fraude com um modelo analítico é identificar possíveis preditores de fraude
associados a criminosos conhecidos e suas ações no passado.

> De acordo com Bolton e Hand (2002), a modelagem supervisionada tem a desvantagem de exigir "certeza absoluta" de que
cada evento pode ser classificado com precisão como fraude ou não fraude. Além disso, os autores observam que qualquer
modelo de fraude pode ser usado para detectar apenas os tipos de fraude que foram identificados anteriormente.

> Métodos não-supervisionados de modelagem de fraude dependem da detecção de eventos anormais. Esses eventos anormais
devem ser caracterizados relacionando os eventos a sintomas associados a eventos fraudulentos no passado.

> Análise de links é o método não supervisionado mais comum de detecção de fraudes. O processo de execução da análise de
link é conhecido como descoberta de link.

> O trabalho de Bolton e Hand (2002) usa a Análise de Grupo de Pares e a Análise de Ponto de Interrupção aplicada ao
comportamento de gastos em contas de cartão de crédito.

> A Análise de Grupo de Pares detecta objetos individuais que começam a se comportar de maneira diferente dos objetos
aos quais eram semelhantes anteriormente.

> Um ponto de interrupção é uma observação em que um comportamento anômalo para uma determinada conta é detectado.
"""
#%%
#  Segmentação de clientes e detecção de anomalias
"""
Contextualização
> A segmentação de clientes é a prática de dividir os clientes de uma empresa em grupos que refletem a semelhança entre
os clientes de cada grupo.

> A detecção de anomalias (também conhecida como análise de outliers) é uma etapa na mineração de dados que identifica
pontos de dados, eventos e/ou observações que se desviam do comportamento normal de um conjunto de dados.

Segmentação de clientes
> O objetivo da segmentação de clientes é decidir como se relacionar com os clientes em cada segmento, a fim de
maximizar o valor de cada cliente para o negócio.

> A segmentação de clientes é o processo de dividir os clientes em grupos com base em características comuns para que
as empresas possam comercializar para cada grupo de forma eficaz e adequada.

> NO marketing B2B uma empresa pode segmentar clientes de acordo com uma ampla gama de fatores, incluindo:
    >> Indústria
    >> Número de empregados
    >> Produtos adquiridos anteriormente da empresa
    >> Localização

Detecção de anomalias
> Dados anômalos podem indicar incidentes críticos, como uma falha técnica, ou oportunidades potenciais, por exemplo,
uma mudança no comportamento do consumidor.

> A detecção bem sucedida de anomalias depende da capacidade de analisar dados de séries temporais com precisão em
tempo real.

> Dependendo do seu modelo de negócios e caso de uso, a detecção de anomalias de dados de séries temporais pode ser
usada para métricas valiosas, como:
    >> Visualizações de páginas da WEB
    >> Usuários ativos diariamente
    >> Instalação de aplicativos móveis
    >> Custo por clique
    >> Custos de aquisição de clientes
    >> Taxa de rotatividade
    >> Receita por clique
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
#%%
# Gerando dados de exemplo
np.random.seed(42)
X = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]
#%%
# Criando o modelo Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)
#%%
# Identificando anomalias
y_pred = clf.predict(X)
outliers = X[y_pred == -1]
#%%
# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='white', edgecolors='k', s=20, label='Data points')
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', edgecolors='k', s=50, label='Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Detecção de Anomalias com Isolation Forest')
plt.legend()
plt.show()
#%%
# Imprimindo os pontos identificados como anomalias
print("Pontos identificados como anomalias:")
print(outliers)