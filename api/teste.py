import mlflow

# 1️⃣ Ajuste para o container correto se estiver rodando via Docker
mlflow.set_tracking_uri("http://localhost:5000") # se rodando local
# mlflow.set_tracking_uri("http://mlflow:5000")  # se rodando dentro do container da API

# 2️⃣ Listar experimentos
experiments = mlflow.get_experiment('8df655085c6043e8a90158938ddf05a6')

for exp in experiments:
    print(f"ID: {exp.experiment_id}, Nome: {exp.name}, Status: {exp.lifecycle_stage}")
