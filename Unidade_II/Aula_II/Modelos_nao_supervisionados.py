# Modelos não supervisionados: definição
"""
Modelos não supervisionados são um tipo de algoritmos de machine learning que operam em dados não rotulados.
Eles são projetados para identificar padrões e estruturas intrínsecas nos dados sem a necessidade de intervenção ou
orientação humana. Aqui estão alguns pontos-chave sobre modelos não supervisionados:

- Descoberta de Padrões: Modelos não supervisionados exploram os dados para encontrar padrões ou agrupamentos naturais¹.
- Clustering: Uma tarefa comum é o agrupamento (clustering), onde o algoritmo organiza os dados em grupos com base em
semelhanças¹.
- Redução de Dimensionalidade: Outra aplicação é a redução de dimensionalidade, que simplifica os dados mantendo apenas
as características mais importantes¹.
- Associação: Modelos não supervisionados também podem ser usados para encontrar regras de associação em grandes
conjuntos de dados¹.

Esses modelos são úteis em situações onde os dados não têm rótulos pré-definidos, ou seja,
não sabemos as respostas ou categorias dos dados de antemão. Eles são ideais para análise exploratória de dados,
segmentação de clientes, reconhecimento de imagem, entre outros¹².
"""

# Modelos não supervisionados – Tipos
"""
1. **Clusterização (Clustering)**:
    - A **clusterização** é uma técnica que agrupa dados com base em suas semelhanças ou diferenças.
    - Algoritmos de clusterização processam objetos de dados não classificados e brutos, agrupando-os com base em
    estruturas ou padrões nas informações.
    - Existem vários tipos de algoritmos de clusterização:
        - **K-Médias (K-Means)**: Atribui pontos de dados a grupos (clusters) com base na distância do centroide de
        cada grupo. É comumente usado em segmentação de mercado, análise de documentos e compactação de imagens.
        - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Identifica clusters com base na
        densidade de pontos de dados em uma região.
        - **Hierárquico**: Agrupa dados em uma estrutura hierárquica de árvore, permitindo a análise de subgrupos.
        - **Gaussian Mixture Models (GMM)**: Modela dados como uma mistura de distribuições gaussianas.
    - A clusterização é útil para segmentação de clientes, análise de imagem e muito mais².

2. **Redução de Dimensionalidade**:
    - A **redução de dimensionalidade** simplifica os dados mantendo apenas as características mais importantes.
    - Algoritmos como **Análise de Componentes Principais (PCA)** e **Autoencoders** são usados para reduzir a
    quantidade de variáveis em um conjunto de dados.
    - Isso ajuda a visualizar dados complexos e reduzir o ruído.
    - É aplicado em reconhecimento de padrões, visualização de dados e otimização de modelos.

3. **Associação**:
    - O aprendizado não supervisionado também lida com **associação de dados**.
    - Nesse contexto, os algoritmos buscam identificar relações entre itens frequentemente co-ocorrentes em um
    conjunto de dados.
    - Um exemplo é a regra de associação **Apriori**, usada para encontrar padrões frequentes em transações de compras.

4. **Detecção de Anomalias**:
    - Algoritmos de detecção de anomalias identificam pontos de dados incomuns ou desviantes.
    - Esses pontos podem representar fraudes, erros ou eventos raros.
    - Exemplos incluem **Isolation Forests** e **One-Class SVM**.

5. **Modelos de Variáveis Latentes**:
    - Esses modelos buscam representar dados em um espaço latente de menor dimensão.
    - **Autoencoders** e **Variational Autoencoders (VAEs)** são exemplos.
    - São usados para compressão de dados, geração de imagens e reconstrução de informações.
"""