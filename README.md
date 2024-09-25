Multi_Modal_Beam_Predict_Pytorch
==============================

>> A comunicação veículo-veículo (V2V) é fundamental em sistemas de transporte inteligente (ITS) para permitir que os veículos troquem informações críticas, aumentando a segurança, a eficiência do tráfego e a experiência de direção em geral. No entanto, métodos tradicionais de comunicação V2V enfrentam dificuldades para lidar com o crescente volume e complexidade de dados, o que pode limitar a eficácia dos ITS.
>>Para enfrentar esse desafio, a exploração de bandas de frequência mais altas, como ondas milimétricas (mmWave) e sub-terahertz (sub-THz), está se tornando cada vez mais crucial. Essas bandas de frequência mais altas oferecem larguras de banda maiores, apresentando uma solução adequada para atender às demandas crescentes de taxa de dados inerentes aos sistemas de comunicação V2V.

No entanto, essa transição traz seus próprios desafios. Os sistemas de alta frequência precisam implantar grandes conjuntos de antenas nos transmissores e/ou receptores e usar feixes estreitos para garantir potência suficiente no receptor. Encontrar os melhores feixes (de um codebook pré-definido) no transmissor e receptor está associado a um alto custo de treinamento de feixe (tempo de busca para encontrar/alinhar os melhores feixes). Esse desafio, portanto, representa um gargalo crítico, especialmente para aplicações V2V de alta mobilidade e sensíveis à latência, marcando a próxima fronteira na comunicação V2V: previsão de feixes V2V eficaz e eficiente.

A previsão de feixes assistida por sensores é uma solução promissora: A dependência de sistemas de comunicação mmWave/THz em links de LOS (linha de visada) entre o transmissor/receptor significa que a consciência sobre suas localizações e o ambiente circundante (geometria dos edifícios, dispersores em movimento, etc.) poderia potencialmente ajudar no processo de seleção de feixes. Para isso, as informações sensoriais do ambiente e do UE podem ser utilizadas para orientar a gestão de feixes e reduzir significativamente o custo de treinamento de feixes. Por exemplo, os dados sensoriais coletados por câmeras RGB, LiDAR, radares, receptores GPS, etc., podem permitir que o transmissor/receptor decidam para onde direcionar seus feixes (ou pelo menos reduzir as direções candidatas de direcionamento de feixes).




Project Organization
------------
    
    ├── LICENSE
    ├── Makefile           <- Makefile com comandos como `make data` ou `make train`
    ├── README.md          <- O README principal para desenvolvedores que usam este projeto.
    ├── Articles           <- Artigos usados como referência para o trabalho.
    ├── data
    │   ├── external       <- Dados de fontes externas.
    │   ├── interim        <- Dados intermediários que foram transformados.
    │   ├── processed      <- Os conjuntos de dados finais e padronizados para modelagem.
    │   └── raw            <- O conjunto de dados original e imutável.
    │
    ├── docs               <- Projeto padrão Sphinx; veja sphinx-doc.org para mais detalhes.
    │
    ├── models             <- Modelos treinados e serializados, predições de modelos ou resumos de modelos.
    │
    ├── notebooks          <- Notebooks Jupyter. Convenção de nomeação é um número (para ordenação),
    │                         as iniciais do criador, e uma breve descrição, e.g., `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Dicionários de dados, manuais e outros materiais explicativos.
    │
    ├── requirements.txt   <- O arquivo de requisitos para reproduzir o ambiente de análise, e.g.,
    │                         gerado com `pip freeze > requirements.txt`.
    │
    ├── setup.py           <- Torna o projeto instalável via pip (pip install -e .) para que o src possa ser importado.
    ├── src                <- Código fonte para uso neste projeto.
    │   ├── __init__.py    <- Torna o src um módulo Python.
    │   │
    │   ├── data           <- Scripts para baixar ou gerar dados.
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts para transformar dados brutos em features para modelagem.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts para treinar modelos e usar modelos treinados para fazer predições.
    │   │   │                 
    │   │   ├── GRU.py  --> Aplicação do modelo GRU.
    │   │   └── cnn.py  --> Aplicação do modelo CNN.
    |   |   ├── baseline.py
    │   │   └── func.py --> Funções complementares que auxiliam a baseline.
    │   │   └── utils.py --> Criação da estrutura dos modelos neurais trabalhados.
    │   │   └── train_model.py --> Funções de treinamento, teste e predições.
    │   │
    │   │
    │   └── visualization  <- Scripts para criar visualizações exploratórias e orientadas para resultados.
    │       └── visualize.py
    │
    └── tox.ini            <- Arquivo tox com configurações para executar tox; veja tox.readthedocs.io.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
