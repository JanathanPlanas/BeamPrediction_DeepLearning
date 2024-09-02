import matplotlib.pyplot as plt

if False:
    # Dados de perda (loss) para as duas unidades GPS durante 50 épocas
    epochs = list(range(1, 51))  # Número de épocas de 1 a 50

    # Loss para a Unidade GPS 1 (completado)
    loss_unit_1 = [
        1.9311657627562417, 0.4661952027652655, 0.4108099444191057, 0.37915818391623807, 
        0.3312412676596521, 0.33595711506163156, 0.2890226209092203, 0.31298206780703014,
        0.28950010787038694, 0.28473307538393444, 0.2847597157836408, 0.2931254794456037,
        0.2796116511134984, 0.27040130965494885, 0.23518619913381783, 0.22056444535852018,
        0.21768300909734242, 0.24696245002793882, 0.2713527310304316, 0.24130090543254873,
        0.26592420483898177, 0.2533423430671738, 0.2332198290226556, 0.23624648848423938,
        0.2680203460210579, 0.2501385632764841, 0.2467195837623034, 0.2423152957134421,
        0.2386094189743298, 0.2350049728201499, 0.2325860759498872, 0.2304523786382184,
        0.2293928791053975, 0.2267009983601983, 0.2251148590271028, 0.2239243510021812,
        0.222300882582731, 0.2217898920183089, 0.2214569807251402, 0.2208725648433293,
        0.2204541232384223, 0.2201276728473898, 0.2198547320444311, 0.2197324701896724,
        0.2195443220238857, 0.219391840158456, 0.2192658927529381, 0.2191483902343125,
        0.2190514629749321, 0.2189652981095937
    ]

    # Loss para a Unidade GPS 2 (completado)
    loss_unit_2 = [
        1.8060361678322583, 0.43170882730952476, 0.38877755109184275, 0.30833198494449926, 
        0.2951840355638216, 0.2736999287654436, 0.26039417363922074, 0.2675388482973534,
        0.238878296066514, 0.24452923025366982, 0.23726886499538333, 0.21600364475276568,
        0.22744764289474684, 0.2153186869918439, 0.22108444410446582, 0.18465583691617754,
        0.18390498887974865, 0.18268255817117443, 0.1725775085178503, 0.17514464882648165,
        0.17643604761786996, 0.1738307498973508, 0.17011648913802924, 0.1707188758187601,
        0.17445573234494355, 0.1705613210992112, 0.169991011837619, 0.1695261035279538,
        0.1691279152401893, 0.1687863946245187, 0.1684891161281139, 0.1682243712854627,
        0.1679867220673534, 0.1677746218301515, 0.1675852150188328, 0.1674162789843253,
        0.167266066385216, 0.1671329125361933, 0.1670153629049146, 0.1669121209930953,
        0.1668219039184785, 0.166743349684015, 0.1666751601849991, 0.166616036225086,
        0.1665648937856197, 0.1665206702991265, 0.1664823958746683, 0.1664492053890763,
        0.1664202250412859, 0.1663946556192102
    ]

    # Plotando o gráfico aprimorado
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, loss_unit_1, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='Unidade GPS 1 Loss', alpha=0.8)
    plt.plot(epochs, loss_unit_2, marker='x', linestyle='-', color='#ff7f0e', linewidth=2, markersize=6, label='Unidade GPS 2 Loss', alpha=0.8)

    plt.fill_between(epochs, loss_unit_1, color='#1f77b4', alpha=0.1)
    plt.fill_between(epochs, loss_unit_2, color='#ff7f0e', alpha=0.1)

    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Perda (Loss)', fontsize=14)
    plt.title('Perda de Treinamento ao Longo das Épocas para Unidades GPS', fontsize=16, fontweight='bold')
    plt.xticks(epochs, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Exibir o gráfico
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo do seaborn
sns.set_theme(style='whitegrid')  # Estilo de fundo claro com grade

# Dados de perda (loss) para as duas unidades GPS durante 50 épocas
epochs = list(range(1, 51))  # Número de épocas de 1 a 50

# Loss para a Unidade GPS 1 (completado)
loss_unit_1 = [
    1.9311657627562417, 0.4661952027652655, 0.4108099444191057, 0.37915818391623807, 
    0.3312412676596521, 0.33595711506163156, 0.2890226209092203, 0.31298206780703014,
    0.28950010787038694, 0.28473307538393444, 0.2847597157836408, 0.2931254794456037,
    0.2796116511134984, 0.27040130965494885, 0.23518619913381783, 0.22056444535852018,
    0.21768300909734242, 0.24696245002793882, 0.2713527310304316, 0.24130090543254873,
    0.26592420483898177, 0.2533423430671738, 0.2332198290226556, 0.23624648848423938,
    0.2680203460210579, 0.2501385632764841, 0.2467195837623034, 0.2423152957134421,
    0.2386094189743298, 0.2350049728201499, 0.2325860759498872, 0.2304523786382184,
    0.2293928791053975, 0.2267009983601983, 0.2251148590271028, 0.2239243510021812,
    0.222300882582731, 0.2217898920183089, 0.2214569807251402, 0.2208725648433293,
    0.2204541232384223, 0.2201276728473898, 0.2198547320444311, 0.2197324701896724,
    0.2195443220238857, 0.219391840158456, 0.2192658927529381, 0.2191483902343125,
    0.2190514629749321, 0.2189652981095937
]

# Loss para a Unidade GPS 2 (completado)
loss_unit_2 = [
    1.8060361678322583, 0.43170882730952476, 0.38877755109184275, 0.30833198494449926, 
    0.2951840355638216, 0.2736999287654436, 0.26039417363922074, 0.2675388482973534,
    0.238878296066514, 0.24452923025366982, 0.23726886499538333, 0.21600364475276568,
    0.22744764289474684, 0.2153186869918439, 0.22108444410446582, 0.18465583691617754,
    0.18390498887974865, 0.18268255817117443, 0.1725775085178503, 0.17514464882648165,
    0.17643604761786996, 0.1738307498973508, 0.17011648913802924, 0.1707188758187601,
    0.17445573234494355, 0.1705613210992112, 0.169991011837619, 0.1695261035279538,
    0.1691279152401893, 0.1687863946245187, 0.1684891161281139, 0.1682243712854627,
    0.1679867220673534, 0.1677746218301515, 0.1675852150188328, 0.1674162789843253,
    0.167266066385216, 0.1671329125361933, 0.1670153629049146, 0.1669121209930953,
    0.1668219039184785, 0.166743349684015, 0.1666751601849991, 0.166616036225086,
    0.1665648937856197, 0.1665206702991265, 0.1664823958746683, 0.1664492053890763,
    0.1664202250412859, 0.1663946556192102
]

# Configuração de cores e estilo
plt.figure(figsize=(12, 8))

# Plotando as curvas de perda
sns.lineplot(x=epochs, y=loss_unit_1, marker='o', color='b', linewidth=2.5, markersize=8, label='Unidade GPS 1 Loss', alpha=0.85)
sns.lineplot(x=epochs, y=loss_unit_2, marker='x', color='orange', linewidth=2.5, markersize=8, label='Unidade GPS 2 Loss', alpha=0.85)

# Adicionando preenchimento para tornar as curvas mais visíveis
plt.fill_between(epochs, loss_unit_1, color='b', alpha=0.1)
plt.fill_between(epochs, loss_unit_2, color='orange', alpha=0.1)

# Configuração dos rótulos e título
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Perda (Loss)', fontsize=14)
plt.title('Perda de Treinamento ao Longo das Épocas para Unidades GPS', fontsize=16, fontweight='bold')

# Ajustes finais e exibição
plt.legend(loc='upper right', fontsize=12)
plt.xticks(epochs, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()