import matplotlib.pyplot as plt
import seaborn as sns

# Dados fornecidos para as duas unidades
epochs = list(range(1, 21))  # Número de épocas de 1 a 20



# Loss de treinamento e validação para a Unidade 2
val_loss_unit_1 = [
    60.733931965435204, 45.051415193795435, 34.23923859343338, 26.702460368371515, 19.166556946416, 
    11.628667709075875, 4.09068438260398, 0.011619464564794789, 0.0009416317490146163, 0.0008274346724065317,
    0.0007918654098398106, 0.0007839053364228835, 0.0007829163542906695, 0.0007819448887926649, 0.0007822881496978567,
    0.0007819266071067295, 0.0007812161240901408, 0.000781579994350142, 0.0007807745514414734, 0.0007808832505189117
]

val_loss_unit_2 = [
    52.640200391309016, 38.04514277227994, 30.497906941380993, 22.962536673710265, 15.424605933551131, 
    7.886570315525449, 0.3486240441943037, 0.0027889660770346507, 0.0022704044802531855, 0.0011511636684158944,
    0.0010405030556159608, 0.00038895231530126505, 0.0005857728069933164, 0.0009805372858957934, 0.0008472925247744946,
    0.0008061106647926801, 0.0008848781477667969, 0.0008563295077510187, 0.000789695870866849, 0.0007958635464072248
]

# Configuração de estilo
sns.set_theme(style='whitegrid')

plt.figure(figsize=(12, 8))


sns.lineplot(x=epochs, y=val_loss_unit_1, marker='x', color='orange', linewidth=2.5, markersize=8, label='Unidade 1 - Validation Loss', alpha=0.85)

sns.lineplot(x=epochs, y=val_loss_unit_2, marker='o', color='b', linewidth=2.5, markersize=8, label='Unidade 2 - Validation Loss', alpha=0.85)

# Preenchimento entre as curvas
plt.fill_between(epochs, val_loss_unit_1, color='orange', alpha=0.1)
plt.fill_between(epochs, val_loss_unit_2, color='r', alpha=0.1)

# Ajustes do eixo y
plt.ylim(0, max( max(val_loss_unit_1), max(val_loss_unit_2)))  # Define o limite do eixo y automaticamente
plt.yticks([i for i in range(0, 65, 5)])  # Ajusta os intervalos do eixo y

# Rótulos e título
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Perda (Loss)', fontsize=14)
plt.title('Perda de Treinamento e Validação ao Longo das Épocas para Unidades GRU', fontsize=16, fontweight='bold')

# Configurações finais
plt.legend(loc='upper right', fontsize=12)
plt.xticks(epochs, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
