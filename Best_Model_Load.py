import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configurar o tracking URI relativo para MLFlow (não será necessário se o modelo estiver fora do MLFlow)
mlflow.set_tracking_uri("file:./mlruns")

# Nome do experimento
experiment_name = "exp_projeto_ciclo_3"
mlflow.set_experiment(experiment_name)

# Carregar o dataset
data = pd.read_csv("water_potability.csv")
data = data.dropna()

# Dividir o dataset em treino e teste
X = data.drop("Potability", axis=1)
y = data["Potability"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Caminho para o modelo na pasta raiz
model_path = "./best_model"

# Carregar o modelo diretamente a partir da pasta raiz
loaded_model = mlflow.pyfunc.load_model(model_path)

# Realizar previsões com dados brutos (por exemplo, os dados de teste)
# Como o pipeline foi salvo, a entrada precisa ser passada diretamente para o modelo,
# que já deve cuidar da normalização e outras transformações, se necessário.
test_predictions = loaded_model.predict(x_test)

# Exibir os dados de entrada (X) e as saídas esperadas (y_test) junto com as previsões
print("Dados de Entrada (X - Features de Teste):")
print(x_test.head())  # Exibe as primeiras 5 linhas de X

print("\nSaídas Esperadas (y_test - Potabilidade Esperada):")
print(y_test.head())  # Exibe as primeiras 5 linhas de y_test

print("\nPrevisões do Modelo (y_pred):")
print(test_predictions[:5])  # Exibe as primeiras 5 previsões do modelo

# Calcular e exibir as métricas de desempenho
# Como o modelo já tem pré-processamento interno, não precisamos passar pelo cross-validation manualmente,
# apenas calcular as métricas diretamente com os dados de teste.
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, average='weighted')
recall = recall_score(y_test, test_predictions, average='weighted')
f1 = f1_score(y_test, test_predictions, average='weighted')

# Exibir as métricas
print("\nMétricas do Modelo Carregado:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
