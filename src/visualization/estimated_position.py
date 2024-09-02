import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Simulação de dados que imitam o comportamento observado
np.random.seed(42)  # Seed para reproducibilidade

x = np.arange(25000)  # Índices das amostras

# Gerar padrões de valores verdadeiros
true_values = np.concatenate([np.random.normal(loc, 10, 5000) for loc in [50, 100, 150, 200, 250]])

# Previsões com ruído adicional
predicted_values = true_values + np.random.normal(0, 20, true_values.shape[0])  

# Criação de um DataFrame para facilitar a plotagem
data = pd.DataFrame({'Index': x, 'True': true_values, 'Pred': predicted_values})

# Configuração do gráfico
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Index', y='True', data=data, color='orange', label='true', s=5, alpha=0.6)
sns.scatterplot(x='Index', y='Pred', data=data, color='blue', label='pred', s=5, alpha=0.6)

plt.xlabel('Índice das Amostras')
plt.ylabel('Valor')
plt.title('Comparação entre Previsões e Valores Verdadeiros')
plt.legend()
plt.show()
