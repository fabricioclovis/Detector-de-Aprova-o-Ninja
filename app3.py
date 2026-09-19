import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# PASSO 1: Preparação dos Dados
# ==========================================
# Criando o DataFrame do "Detector de Aprovação Ninja"
alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0] # 1 = Aprovado, 0 = Reprovado
})

# Separando as variáveis
# X (entrada): Faltas do aluno
# y (saída esperada): Resultado (0 ou 1)
X = alunos['faltas'].values
y = alunos['resultado'].values


# ==========================================
# PASSO 2: Criação do Modelo (Regressão Logística)
# ==========================================
# Iniciamos um modelo Sequencial (uma estrutura linear de camadas)
modelo = Sequential()

# Adicionamos uma única camada Densa com ativação 'sigmoid'
# units=1: Queremos apenas 1 valor de saída (a probabilidade de aprovação)
# input_shape=[1]: Temos apenas 1 variável de entrada (quantidade de faltas)
# activation='sigmoid': Transforma qualquer número gerado em um valor entre 0 e 1
modelo.add(Dense(units=1, input_shape=[1], activation='sigmoid'))


# ==========================================
# PASSO 3: Compilação do Modelo
# ==========================================
# loss='binary_crossentropy': Função matemática ideal para problemas de "Sim ou Não" (0 ou 1)
# optimizer=Adam: Ajusta os pesos do modelo para diminuir o erro. Taxa de 0.1 acelera o teste.
# metrics=['accuracy']: Pede para o TensorFlow nos mostrar a taxa de acerto do modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# PASSO 4: Treinamento (Ajuste/Fit)
# ==========================================
print("Treinando o Detector de Aprovação Ninja...")

# Treinamos por 500 épocas. verbose=0 oculta o progresso linha a linha para um terminal limpo.
modelo.fit(X, y, epochs=500, verbose=0)
print("Treinamento concluído com sucesso!\n")


# ==========================================
# PASSO 5: Teste Prático (Predição)
# ==========================================
# Testando o que acontece com um aluno que faltou 4 vezes
faltas_teste = np.array([4.0])

# O modelo preverá um valor entre 0 e 1 (a probabilidade de estar na categoria 1)
previsao = modelo.predict(faltas_teste, verbose=0)
probabilidade = previsao[0][0]

# Definimos o limiar: se a probabilidade for maior ou igual a 0.5 (50%), ele é aprovado
status = "APROVADO" if probabilidade >= 0.5 else "REPROVADO"

print(f"--- RESULTADO DA PREDIÇÃO ---")
print(f"Faltas do aluno: {faltas_teste[0]:.0f}")
print(f"Probabilidade matemática de aprovação: {probabilidade * 100:.1f}%")
print(f"Veredito final: {status}")