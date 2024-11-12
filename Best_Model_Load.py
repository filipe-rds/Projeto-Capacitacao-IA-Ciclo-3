import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Configurar o tracking URI relativo para MLFlow
mlflow.set_tracking_uri("file:./mlruns")

# Nome do experimento
experiment_name = "exp_projeto_ciclo_3"
mlflow.set_experiment(experiment_name)

# Inicializar o cliente do MLFlow
client = MlflowClient()

# Obter o ID do experimento
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

# Carregar o dataset
data = pd.read_csv("water_potability.csv")

# Dividir o dataset em treino e teste
X = data.drop("Potability", axis=1)
y = data["Potability"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Obter todos os runs do experimento, ordenando pelo F1 Score em ordem decrescente
runs = client.search_runs(experiment_id, order_by=["metrics.F1 Score DESC"], max_results=1)

# Verificar se há runs disponíveis
if runs:
    # Selecionar o melhor run
    best_run = runs[0]
    best_model_name = best_run.data.tags.get("mlflow.runName")  # Nome do modelo pelo run name

    # Carregar o modelo usando o URI relativo
    model_uri = f"runs:/{best_run.info.run_id}/{best_model_name}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Realizar o teste no modelo carregado
    test_predictions = loaded_model.predict(x_test)

    # Calcular e exibir as métricas de desempenho
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, average='weighted')
    recall = recall_score(y_test, test_predictions, average='weighted')
    f1 = f1_score(y_test, test_predictions, average='weighted')

    print("Previsões do modelo carregado:", test_predictions)
    print("\nMétricas do modelo carregado:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("Nenhum run encontrado para o experimento especificado.")
