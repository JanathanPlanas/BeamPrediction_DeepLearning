Multi_Modal_Beam_Predict_Pytorch
==============================

>> A comunicação veículo-veículo (V2V) é fundamental em sistemas de transporte inteligente (ITS) para permitir que os veículos troquem informações críticas, aumentando a segurança, a eficiência do tráfego e a experiência de direção em geral. No entanto, métodos tradicionais de comunicação V2V enfrentam dificuldades para lidar com o crescente volume e complexidade de dados, o que pode limitar a eficácia dos ITS.
>>Para enfrentar esse desafio, a exploração de bandas de frequência mais altas, como ondas milimétricas (mmWave) e sub-terahertz (sub-THz), está se tornando cada vez mais crucial. Essas bandas de frequência mais altas oferecem larguras de banda maiores, apresentando uma solução adequada para atender às demandas crescentes de taxa de dados inerentes aos sistemas de comunicação V2V.

No entanto, essa transição traz seus próprios desafios. Os sistemas de alta frequência precisam implantar grandes conjuntos de antenas nos transmissores e/ou receptores e usar feixes estreitos para garantir potência suficiente no receptor. Encontrar os melhores feixes (de um codebook pré-definido) no transmissor e receptor está associado a um alto custo de treinamento de feixe (tempo de busca para encontrar/alinhar os melhores feixes). Esse desafio, portanto, representa um gargalo crítico, especialmente para aplicações V2V de alta mobilidade e sensíveis à latência, marcando a próxima fronteira na comunicação V2V: previsão de feixes V2V eficaz e eficiente.

A previsão de feixes assistida por sensores é uma solução promissora: A dependência de sistemas de comunicação mmWave/THz em links de LOS (linha de visada) entre o transmissor/receptor significa que a consciência sobre suas localizações e o ambiente circundante (geometria dos edifícios, dispersores em movimento, etc.) poderia potencialmente ajudar no processo de seleção de feixes. Para isso, as informações sensoriais do ambiente e do UE podem ser utilizadas para orientar a gestão de feixes e reduzir significativamente o custo de treinamento de feixes. Por exemplo, os dados sensoriais coletados por câmeras RGB, LiDAR, radares, receptores GPS, etc., podem permitir que o transmissor/receptor decidam para onde direcionar seus feixes (ou pelo menos reduzir as direções candidatas de direcionamento de feixes).




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── Articles       <- Artigos usados como referencia para o trabalho.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── GRU.py  --> Aplicação do modelo GRU
    │   │   └── cnn.py  --> Aplicação do modelo CNN
    |   |   ├── baseline.py
    │   │   └── func.py --> Funções complementares que auxiliam a baseline
    │   │   └── utils.py --> Criação da estrutura dos modelos neurais trabalhados
    │   │   └── train_model.py --> Funções de treinamento , teste e predições
    │   │
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
