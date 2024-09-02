import plotly.graph_objects as go
import pandas as pd

# Dados de exemplo
data = {
    'Coluna 1': [10, 20, 30],
    'Coluna 2': [15, 25, 35],
    'Coluna 3': [20, 30, 40]
}
df = pd.DataFrame(data)

# Cria a tabela com Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='center',
                font=dict(color='black', size=12)),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='lavender',
               align='center',
               font=dict(color='black', size=11))
)])

# Salva a figura como PNG
fig.write_image("tabela_plotly.png")
