import matplotlib.pyplot as plt

# Dados da APL
labels = ['Treino', 'Teste']
apl_values = [-4.81, -7.81]  # Valores de APL para treino e teste

# Criação do gráfico
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, apl_values, color=['#1f77b4', '#ff7f0e'], alpha=0.85, edgecolor='black', linewidth=1.2)

# Adiciona a linha de referência em y=0
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

# Configuração dos detalhes do gráfico
plt.title('Average Power Loss (APL) - Treino vs Teste', fontsize=14, fontweight='bold')
plt.ylabel('Average Power Loss (APL) [dB]', fontsize=12)
plt.xlabel('Dataset', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(-10, 0)  # Limites do eixo y

# Adição de anotações dos valores de APL
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} dB', ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

# Melhorando a estética do gráfico
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

plt.show()
