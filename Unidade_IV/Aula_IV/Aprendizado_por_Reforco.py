# Aprendizado por Reforço – Definição
"""
Contextualização
> Aprendizado por reforço
    >> Tentativa e erro.
    >> Forma de aprendizado dos seres humanos, que podem aprender baseados em suas experiências anteriores.

Aprendizado por reforço
> Idéia básica:
 >> Um agente procura cumprir uma determinada tarefa.
 >> Sistema de recompensa e punição.
 >> Abordagem de tentativa e erro.

> Agente:
    >> Entidade que toma decisões.
> Ambiente:
    >> Espaço no qual o agente realiza suas ações.
> Estado(s):
    >> Sistema formado pelo agente e pelo ambiente, em um determinado instante de tempo.
"""
#%%
# Aprendizado por Reforço – Modelo
"""
Contextualização
> Aprendizado por reforço
    >> Técnica que permite que um agente tome ações e interaja com um ambiente.

Aprendizado por reforço
> Política
    >> Estratégia aplicada pelo agente para decidir a próxima ação, com base no estado atual.
> Recompensa
    >> Orienta as ações do agnete, sendo a política ideal determinada a partir dela.
> MDP
    >> Processo de decisão de Markov.
    >> Lógica do aprendizado por reforço.

MDP (Markov Decision Process)
>> Agente >> Ação At >> Ambiente >> Recompensa Rt+1 >> Agente >>
                                >> Estado St+1
"""
#%%
import numpy as np

# Define os estados do ambiente
states = ['estado1', 'estado2', 'estado3']

# Define as ações que o agente pode tomar em cada estado
actions = ['ação1', 'ação2', 'ação3']

# Define as transições de estado e as recompensas associadas a cada ação em cada estado
transition_probabilities = np.array([
    [[0.7, 0.1, 0.2], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]],  # Transições de estado a partir de 'estado1'
    [[0.4, 0.3, 0.3], [0.6, 0.1, 0.3], [0.2, 0.4, 0.4]],  # Transições de estado a partir de 'estado2'
    [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.6, 0.15, 0.25]]  # Transições de estado a partir de 'estado3'
])

rewards = np.array([
    [+1, -1, +0],  # Recompensas associadas a cada ação em 'estado1'
    [-1, +0, +1],  # Recompensas associadas a cada ação em 'estado2'
    [+0, +1, -1]   # Recompensas associadas a cada ação em 'estado3'
])

# Define a política do agente
def policy(state):
    if state == 'estado1':
        return 'ação1'
    elif state == 'estado2':
        return 'ação2'
    else:
        return 'ação3'

# Executa o MDP
state = 'estado1'
total_reward = 0

for _ in range(10):
    action = policy(state)
    next_state = np.random.choice(states, p=transition_probabilities[states.index(state), actions.index(action)])
    reward = rewards[states.index(state), actions.index(action)]
    total_reward += reward
    state = next_state

print("Total recompensa:", total_reward)
#%%
# Aprendizado por Reforço – Aplicação
"""
Contextualização
> Aprendizado por reforço:
    >> Estrutura matemática para resolução de problemas.
    
Aprendizado por Reforço
> Gerenciamento de Recursos:
    >> Calculo de Preços
> Controle de Semáforo:
    >> Circulação de Automóveis
> Configuração de Sistemas Web:
    >> Criação e gestão de recursos de computação.
> Química:
    >> Cálculo de Equação de Massa Molecula
> Jogos:
    >> Bots para jogos
"""