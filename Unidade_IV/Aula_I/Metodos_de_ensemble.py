# Modelos de Ensemble - Definição
"""
Contextualização
> Abundância de estimadores de Machine Learning.
> Esses estimadores possuem vantagens e desvantagens.

Ensemble Learning
> União de diversos modelos de predição mais simples.
> Obtenção de um modelo geral.
> Modelo mais consistente e menos suscetível a ruídos.
> Chamado de aprendizado de agrupamento.

Objetivo:
> Diminuição do BIAS e Variance
> Exemplo: classificação binária

Agrupamento:
> Construção de modelos.
> Escolher apenas um modelo base
> Realizar combinação;
> Exemplo: Escolher o modelo de melhor desempenho.
> Ensemble Learning: Combinação de métodos.
"""
#%%
# Modelos de Ensemble - Tipos
"""
Contextualização
> Ensembles:
    >> Combinações de diferentes modelos fracos para a criação de um modelo mais forte.
    >> Produzem melhores resultados.
    >> Existem vários tipos de abordagem para agrupamento.
    
Ensemble Learning - Tipos
> Stacking
    >> Cross-validation.
    >> Preditores heterogêneos.
    >> Treinamento em paralelo.
> Bagging
    >> Evitar overfitting.
    >> Resultado final: média das respostas.
    >> Treinamento em paralelo.
> Boosting
    > Preditores homogêneos.
    > Treinamento sequencial.
    > Resultado final: combinação.
    
Árvore de Decisão
> Uma das formas mais simples de um sistema de suporte à decisão.
> Método estatístico.
> Aprendizagem supervisionada.
> Aplicações: classificação ou previsão.
> Conjunto de dados existente:
    >> Cria-se uma representação do conhecimento ali embutido;
    >> Formato de Árvore;
> Elementos:
    >> Nós = características;
    >> Raiz = início;
> Mapa de possíveis resultados.
> Comparação de custos e benefícios.
> Geralmente composta por perguntas e respostas booleanas.

Random Forest
> Algoritmo de agrupamento.
> Cria uma floresta aleatória.
> São criadas várias árvores de decisão que juntas geram uma resposta final.
> Utiliza a Moda (mode) como previsão.
"""
#%%
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
#%%
# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target
#%%
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Criando o modelo Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
#%%
# Prevendo as classes das amostras de teste
y_pred = clf.predict(X_test)
#%%
# Avaliando o modelo
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(cm)
print("\nRelatório de classificação:")
report = classification_report(y_test, y_pred, zero_division=1)
print(report)
#%%
# Calculando o coeficiente de correlação e o p-valor
correlation, p_value = pearsonr(y_test, y_pred)
print("Coeficiente de correlação:", correlation)
print("p-valor:", p_value)
#%%
# Plotando os dados
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=50)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k', s=50, alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest - Classificação das especies de Iris')
plt.colorbar()
plt.show()
#%%
# Modelos de Ensemble - Exemplos
"""
Contextualização
> Essemble Learning:
    >> Combinação de modelos diferentes para a obtenção de um único resultado.
    >> Robustos.
    >> Maior custo computacional.
    >> Melhores resultados.
    
Maneiras de Aplicação
> Estatística Simples
    >> Problemas de classificação:
        >>> Esquema de votação.
        >>> Por exemplo: medida de moda.
    >> Problemas de regressão:
        >>> Por exemplo: média ou mediana.
        
Criação do Modelo
> Escolha do algoritmo que apresenta o melhor desempenho para os dados em questão.
> Teste de diferentes configurações.
> Geração de diferentes modelos.
> Agregação de resultados.

Aplicações
> Problemas de classificação:
    >> Esquema de votação.
> Problemas de regressão:
    >> Média dos votos de cada árvore.
> Várias formas que podem ser usadas como métodos ensemble para agregar o resultado de diferentes modelos separados.
> Exemplos:
    >> Classificação de clientes inadimplentes em operações de crédito pessoal;
    >> Análise de fraudes;
    >> Reconhecimento facial;
    >> Previsão de rotas rodoviárias;            
"""