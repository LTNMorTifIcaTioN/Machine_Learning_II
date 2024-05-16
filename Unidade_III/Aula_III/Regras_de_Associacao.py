# Regras de associação – Definição
"""
Contextualização:
> As regras de associação são abordagens de mineração de dados usadas para explorar e interpretar grandes conjuntos de
dados transacionais para identificar padrões e regras exclusivas.

Definição:
> As regras ajudam a identificar e prever comportamentos transacionais com base nas informações de transações
de treinamentos que utilizam propriedades benéficas.

> São tipos de técnicas ed aprendizado não supervisionado que verificam a dependência entre dados, ou seja, tentam
encontrar algumas relações ou associações interessantes entre as variáveis do conjunto de dados.

> São bastante utilizadas para descobrir correlações entre produtos em dados de transações de grande escala registrados
por sistemas de ponto de vendas em supermercados

> Uma das premissas da regra de associação é a ideia que um ou mais itens juntos são associados a outro que compartilha
algo em comum.

Exemplo: {Cebolas + Batatas} = {Hambúrguer}
"""
#%%
# Regras de associação – Aplicação 1
"""
Contextualização:
> Uma regra de associação tem 2 partes: Um antecedente (se) e um consequente (então). Um antecedente é algo encontrado
nos dados e um consequente é um item encontrado em combinação com o antecedente.

Aplicação:
> Dependendo dos dois parâmetros,as relações importantes são observadas:
    1. Suporte: O suporte indica com que frequência a relação se/então aparece no banco de dados;
    2. Confiança: A confiança fala sobre o número de vezes que essas relações foram consideradas verdadeiras;
    
> A confiança de uma regra representa sua força e define-se em:
    Confiança(x,y) = (Suporte(x,y) / Suporte(x)) * 100%
"""
#%%
# Regras de associação – Aplicação 2
"""
Contextualização:
> Suporte e confiança são medidas de confiabilidade e frequência de ocorrência.

"""
#%%
# Exemplo itens de praia:
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Criando um DataFrame de exemplo
data = {'Maiôs': [1, 1, 1, 0],
        'Boné': [1, 0, 0, 1],
        'Óculos de sol': [1, 0, 0, 0],
        'Toalha': [1, 1, 0, 1],
        'Bermuda': [0, 0, 1, 0],
        'Camisa': [0, 0, 0, 1]}
df = pd.DataFrame(data)

# Calculando os itens frequentes com o algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# Calculando as regras de associação
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)

# Exibindo as regras encontradas
rules[['antecedents', 'consequents', 'support', 'confidence']]
#%%
rules_antecedents = rules[rules['antecedents'] == {'Maiôs', 'Toalha'}]
rules_consequents = rules_antecedents[rules_antecedents['consequents'] == {'Óculos de sol'}]
rules_consequents
