# Importando a biblioteca pandas para manipulação de dados
import pandas as pd
# Importando a biblioteca numpy para operações numéricas
import numpy as np
# Importando a biblioteca matplotlib.pyplot para visualização de dados
import matplotlib.pyplot as plt
# Importando a biblioteca seaborn para visualização estatística de dados
import seaborn as sns
# Importando a classe PCA (Análise de Componentes Principais) do scikit-learn
from sklearn.decomposition import PCA
# Importando a classe StandardScaler para normalização de dados do scikit-learn
from sklearn.preprocessing import StandardScaler

################# # 1. Coleta de Dados # ######################

# Define uma semente para o gerador de números aleatórios para garantir a reprodutibilidade dos resultados.
np.random.seed(0)
# Cria uma sequência de datas de venda ao longo de seis meses, começando em 01/01/2023 e terminando em 30/06/2023, com uma frequência diária.
datas_vendas = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
# Define uma lista de produtos disponíveis para venda.
produtos = ['Produto A', 'Produto B', 'Produto C']
# Define uma lista de clientes que podem fazer compras.
clientes = ['Cliente 1', 'Cliente 2', 'Cliente 3']
# Cria um dicionário contendo os dados fictícios de vendas, onde cada chave é o nome de uma coluna e cada valor é uma série de dados correspondente.
dados = {
    'Data_Venda': np.random.choice(datas_vendas, 100),  # Seleciona aleatoriamente 100 datas de venda da sequência de datas.
    'Produto': np.random.choice(produtos, 100),         # Seleciona aleatoriamente 100 produtos da lista de produtos.
    'Quantidade': np.random.randint(1, 20, 100),        # Gera aleatoriamente 100 números inteiros entre 1 e 20 para representar a quantidade vendida.
    'Valor_Venda': np.random.uniform(50, 200, 100),     # Gera aleatoriamente 100 números decimais uniformemente distribuídos entre 50 e 200 para representar o valor de cada venda.
    'Cliente': np.random.choice(clientes, 100)         # Seleciona aleatoriamente 100 clientes da lista de clientes.
}
# Cria um DataFrame pandas a partir do dicionário de dados, onde cada chave do dicionário se torna uma coluna no DataFrame.
df_vendas = pd.DataFrame(dados)

# Salvando os dados em um arquivo Excel
df_vendas.to_excel("dados_vendas.xlsx", index=False)

# 2. Limpeza e Preenchimento de Dados
# Não é necessário nesta etapa para este exemplo

# 3. Transformação de Dados
# Não é necessário nesta etapa para este exemplo

########################## # 4. Exploração de Dados  ###############################
# Esta linha de código calcula os descritores estatísticos para cada coluna do DataFrame 'df_vendas' usando o método 'describe()'.
# Os descritores estatísticos incluem contagem, média, desvio padrão, valor mínimo, percentis (25%, 50%, 75%) e valor máximo.
# O resultado é armazenado na variável 'descritores_estatisticos'.
descritores_estatisticos = df_vendas.describe()

# Esta linha de código imprime uma mensagem indicando que os descritores estatísticos serão exibidos.
print("Descritores Estatísticos:")
print(descritores_estatisticos)

########################## # 5. Visualização de Dados #############################
# Histograma de valores de venda
# Esta linha de código cria uma nova figura para o gráfico com uma largura de 10 polegadas e altura de 6 polegadas.
plt.figure(figsize=(10, 6))

# Esta linha de código utiliza a função histplot da biblioteca Seaborn para criar um histograma dos valores de venda.
# Passamos a coluna 'Valor_Venda' do DataFrame 'df_vendas' como dados para o histograma.
# O argumento 'kde=True' adiciona uma estimativa de densidade de kernel à plotagem, suavizando a distribuição.
sns.histplot(df_vendas["Valor_Venda"], kde=True)

# Este comando define o título do gráfico como "Distribuição de Valores de Venda".
plt.title("Distribuição de Valores de Venda")

# Este comando define o rótulo do eixo x como "Valor de Venda".
plt.xlabel("Valor de Venda")

# Este comando define o rótulo do eixo y como "Contagem", indicando que o eixo y representa a contagem de observações em cada intervalo de valor.
plt.ylabel("Contagem")

# Este comando exibe o gráfico na tela.
plt.show()

######################## # 6. Redução de Dados e Análise por Componentes Principais ###################################
# Cria uma instância da classe StandardScaler do scikit-learn, que é usada para normalizar os dados.
# A normalização é uma técnica comum de pré-processamento que ajusta os dados para ter uma média de zero e um desvio padrão de um.
scaler = StandardScaler()

# Utiliza o método fit_transform() do objeto scaler para normalizar os dados do DataFrame 'df_vendas'.
# Apenas as colunas 'Quantidade' e 'Valor_Venda' são selecionadas para a normalização usando a sintaxe de indexação ['Quantidade', 'Valor_Venda'].
# O método fit_transform() ajusta o scaler aos dados e, em seguida, transforma os dados, retornando os dados normalizados.
# Os dados normalizados são atribuídos à variável 'dados_normalizados' para uso posterior.
dados_normalizados = scaler.fit_transform(df_vendas[['Quantidade', 'Valor_Venda']])

# Cria uma instância da classe PCA (Principal Component Analysis) do scikit-learn.
# Define o número de componentes principais desejados como 2 utilizando o argumento n_components=2.
# O PCA é uma técnica de redução de dimensionalidade que transforma os dados originais em um novo conjunto de variáveis (componentes principais) que são combinações lineares dos recursos originais.
pca = PCA(n_components=2)

# Utiliza o método fit_transform() do objeto PCA para ajustar o modelo aos dados normalizados e, em seguida, transformar os dados.
# Os dados normalizados (dados_normalizados) são passados como entrada para o método fit_transform().
# O método fit_transform() ajusta o PCA aos dados, calculando os componentes principais e, em seguida, transforma os dados originais em um novo conjunto de dados reduzidos.
# Os dados reduzidos são atribuídos à variável 'dados_reduzidos' para uso posterior.
dados_reduzidos = pca.fit_transform(dados_normalizados)


# Cria um novo DataFrame pandas chamado 'df_reduzido'.
# O argumento 'data' especifica os dados que serão usados para construir o DataFrame, que são os dados reduzidos após aplicar o PCA.
# 'dados_reduzidos' contém as coordenadas dos pontos de dados no novo espaço de características após a redução de dimensionalidade.
# O argumento 'columns' especifica os nomes das colunas do DataFrame, que são 'Componente 1' e 'Componente 2', representando as duas dimensões resultantes após a redução.
# O DataFrame 'df_reduzido' será usado posteriormente para visualizar os dados em um gráfico de dispersão ou em outras análises.
df_reduzido = pd.DataFrame(data=dados_reduzidos, columns=['Componente 1', 'Componente 2'])


# Visualizando os dados reduzidos
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Componente 1', y='Componente 2', data=df_reduzido)
plt.title("Análise por Componentes Principais")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()