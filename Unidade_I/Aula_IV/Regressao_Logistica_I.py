# Método de classificação: uma introdução
"""
Curva Roc
> A curva das características operacionais do receptor utilizada com classificadores binários.
> É semelhante à curva de precisão/revocação, mas, em vez de plotar a precisão vs revocação, a curva ROC plota a
taxa de verdadeiro positivo (TPR) vs a taxa de falso positivo (FPR).
> FPR é a razão de instâncias negativas classificadas como positivas.
> é 1-taxa de verdadeiro negativo (TNR).
> A TNR é também designada por especificidade (SPC).
> Quanto maior a revocação (TPR), mais falsos positivos (FPR) o classificador produz.
> Um classificador perfeito tera um ROC AUC = 1, enquanto o classificador aleatório terá um ROC AUC = 0,5.
> Usar Curva ROC para Falsos Negativos e curva PR para Falsos Positivos.
"""
#%%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Criar dados aleatórios para testar a curva ROC
np.random.seed(42)
y_true = np.random.randint(2, size=1000)
y_scores = np.random.rand(1000)

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calcular a área sob a curva ROC
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Linha de base')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.legend()
plt.show()
#%%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_scores)
#%%
# Função logística versus Regressão logística
"""
Regressão Logística
> É comumente utilizada para estimar a probabilidade de uma instância pertencer à uma determinada classe.
> Se a probabilidade de uma instância pertencer à classe positiva for maior que 0,5, a instância será classificada como
positiva, portanto rotulada como "1". Caso contrário, será classificada como negativa, portanto rotulada "0".
> Assim como um modelo de Regressão Linear, um modelo de Regressão Logística calcula uma soma ponderada das
características de entrada (mais um termo de polarização), mas em vez de gerar o resultado diretamente como o modelo de
Regressão Linear, gera a logística desse resultado.
> A logística, também chamada de logit, é uma função sigmoidal, que gera valores entre 0 e 1:
    s = 1 / (1 + exp(-z))
    z = x * beta
    logit = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ...
"""
#%%
# Regressão logística: função de predição
"""
Treinamento e Função Custo
> A função custo e o método de treinamento para o modelo de Regressão Logística.
c(t) = {-log(p^) se y = 1, -log(1 - p^) se y = 0}
> O custo será maior se o modelo estimar uma probabilidade próxima a 0 para uma instância positiva
e próxima a 1 para uma instância negativa.
> -log(t) é próximo de 0 quanto t for próximo de 1.
> A função de custo é o em relação ao conjunto de treinamento é o curso médio em relação a todas as instâncias de
treinamento.
Equação Log Loss = = - ∑ (1 / n) ∑ (yᵢ * log(pᵢ) + (1 - yᵢ) * log(1 - pᵢ))

Onde:

n é o número de amostras
yᵢ é a classe verdadeira (0 ou 1) para a amostra i
pᵢ é a probabilidade predita para a classe 1 pelo modelo (pode ser qualquer probabilidade entre 0 e 1) 

A equação das derivadas parciais da função de custo com relação ao j-ésimo modelo do parâmetro tetaJ depende do
algoritmo de aprendizado de máquina que você está utilizando. No entanto, a derivada parcial da função de custo com
relação ao j-ésimo modelo do parâmetro thetaJ é geralmente definida como:

dC/dthetaJ = (1/2) * (y_j - ŷ_j) * x_jj

Onde:

C é a função de custo
thetaJ é o j-ésimo parâmetro do modelo (ou seja, o j-ésimo coeficiente na equação da regressão)
y_j é a classe verdadeira da amostra j
ŷ_j é a classe predita pela função de decisão (ou seja, o resultado do modelo aplicado à amostra j)
x_jj é a j-ésima característica da amostra j
Essa equação é usada para estimar a taxa de aprendizado alpha em algoritmos de aprendizado de máquina como Gradient
Descent. Note que a derivada parcial da função de custo depende do modelo específico e da função de decisão utilizada,
então a equação pode variar dependendo desses fatores.
"""
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Criar dados aleatórios para testar a regressão logística
np.random.seed(42)
X = pd.DataFrame(np.random.rand(1000, 5), columns=['feature_%d' % i for i in range(5)])
y = np.random.randint(2, size=1000)

# Dividir os dados em um conjunto de treinamento e um conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo de regressão logística e treiná-lo
model = LogisticRegression()
model.fit(X_train, y_train)

# Calcular as previsões para o conjunto de teste
y_pred = model.predict(X_test)

# Calcular as probabilidades de classe para o conjunto de teste
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Calcular a curva de precisão-sensibilidade
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plotar as curvas de aprendizado e a curva de precisão-sensibilidade
plt.plot(thresholds, precision[:-1], label='Precisão')
plt.plot(thresholds, recall[:-1], label='Sensibilidade')
plt.xlabel('Limiar')
plt.ylabel('Valor')
plt.title('Curva de Aprendizado')
plt.legend()
plt.show()

# Plotar a curva ROC
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Linha de base')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.legend()
plt.show()
#%%
# Iris Virginica
"""
O profissional que trabalha com análise de dados precisa possuir conhecimento para discernir qual é o modelo mais
apropriado a ser usado. Nesse sentido, existem algumas vantagens de uso do modelo de regressão logística.
Você utilizará a base de dados da Iris para realizar uma classificação de pétalas. A ideia, aqui, é que você consiga
montar um classificador para identificar o tipo Iris-Virginica baseado na característica do comprimento da pétala.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Carregar o dataset Iris
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # 1 se a espécie é Iris-Virginica, 0 caso contrário

# Selecionar apenas as características relacionadas às pétalas (índice 2 e 3)
X_petalas = X[:, 2:4]

# Criar o modelo Logistic Regression
model = LogisticRegression()
model.fit(X_petalas, y)

# Plotar as características das pétalas e a fronteira de decisão do modelo
plt.figure(figsize=(10, 6))
plt.scatter(X_petalas[y==0][:, 0], X_petalas[y==0][:, 1], color='orange', label='Não Iris-Virginica')
plt.scatter(X_petalas[y==1][:, 0], X_petalas[y==1][:, 1], color='blue', label='Iris-Virginica')

# Plotar a fronteira de decisão do modelo
xx = np.linspace(X_petalas[:, 0].min(), X_petalas[:, 0].max())
yy = -(model.coef_[0][0] / model.coef_[0][1]) * xx - (model.intercept_[0] / model.coef_[0][1])
plt.plot(xx, yy, 'k--', color='black', label='Fronteira de Decisão')

plt.xlabel('Comprimento da Pétala')
plt.ylabel('Largura da Pétala')
plt.title('Classificação de Iris-Virginica')
plt.legend()
plt.show()