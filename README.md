README.md — Projeto MLOps: Predição de Rotatividade de Funcionários (Employee Attrition)

Este projeto apresenta uma solução completa de Machine Learning + MLOps para prever a probabilidade de um funcionário deixar a empresa (employee attrition).
Toda a solução foi construída com Python, FastAPI, MLflow e Docker, seguindo boas práticas de modularização, versionamento de modelos e padronização do ambiente.

👨‍🎓 Alunos

Diego Planinscheick

Gabriel Henrique Karing

José Marcos Zoz Marques

📘 Descrição Geral

O sistema foi desenvolvido como parte de um mini-projeto de MLOps e tem como objetivo transformar um notebook de Machine Learning em:

✅ Um pipeline de treinamento
✅ Um modelo versionado com MLflow
✅ Uma API REST de predição
✅ Um ambiente executável via Docker
✅ Uma interface web simples para análise (HTML/CSS/JS)

O modelo principal utiliza Regressão Logística, Árvore de Decisão e Random Forest. O melhor modelo (maior Recall) é automaticamente selecionado e salvo em model.pkl.

📂 Arquitetura do Projeto
/
├── data/
│ └── WA*Fn-UseC*-HR-Employee-Attrition.csv
├── static/
│ └── index.html # Interface Web de Predição
├── app.py # API (FastAPI)
├── train.py # Pipeline de treino + MLflow
├── predict.py # Módulo de inferência
├── model.pkl # Modelo final (gerado automaticamente)
├── Dockerfile # Docker para execução da API
├── requirements.txt # Dependências
└── README.md # Documentação

📊 Dataset Utilizado

Nome: HR Employee Attrition

Fonte: Kaggle — IBM HR Analytics

URL: https://www.kaggle.com/datasets/miraclenifise/hr-employee-attrition-datasets/data

🎯 Variável-alvo

Attrition — se o funcionário deixou a empresa:

Yes → 1

No → 0

🔑 Principais Features Utilizadas
Feature Descrição
DistanceFromHome Distância da casa ao trabalho
EnvironmentSatisfaction Satisfação com ambiente (1–4)
JobSatisfaction Satisfação no trabalho (1–4)
YearsAtCompany Tempo na empresa
WorkLifeBalance Equilíbrio vida–trabalho
PercentSalaryHike Aumento salarial (%)
YearsInCurrentRole Tempo no cargo atual
YearsSinceLastPromotion Anos desde última promoção
🛠️ Pré-requisitos

Para executar localmente:

✔️ Python 3.9+
✔️ Pip / Venv
✔️ Docker Desktop instalado e ativo
🚀 Como Executar o Projeto
1️⃣ Criar ambiente virtual (opcional, recomendado)
python -m venv venv

venv\Scripts\activate # Windows

2️⃣ Instalar dependências
pip install -r requirements.txt

3️⃣ Treinar o modelo (OBRIGATÓRIO antes do Docker)
python train.py

Isso irá:

✔️ Carregar e limpar os dados
✔️ Testar múltiplos modelos
✔️ Registrar métricas no MLflow
✔️ Salvar o melhor modelo como model.pkl

4️⃣ Construir a imagem Docker
docker build -t hr-attrition-api .

5️⃣ Executar o container
docker run -p 8000:8000 hr-attrition-api

API disponível em:

👉 http://localhost:8000

Documentação interativa (Swagger):
👉 http://localhost:8000/docs

🧪 Testando a API
Via Swagger

Acesse http://localhost:8000/docs

Clique em POST /predict

Clique em Try it out

Insira o JSON abaixo:

{
"DistanceFromHome": 10,
"EnvironmentSatisfaction": 1,
"JobSatisfaction": 1,
"YearsAtCompany": 2,
"WorkLifeBalance": 1,
"PercentSalaryHike": 11,
"YearsInCurrentRole": 1,
"YearsSinceLastPromotion": 0
}

🖥️ Interface Web

Este projeto inclui um pequeno frontend localizado em /static/index.html.

Após subir a API, abra:

👉 http://localhost:8000/

Lá você poderá inserir dados e visualizar o nível de risco:

🟢 Baixo

🟡 Médio

🔴 Alto

O frontend usa a própria API /predict.

🔧 Tecnologias Utilizadas
Backend

Python

FastAPI

Uvicorn

Machine Learning

Scikit-Learn

Pandas / NumPy

MLflow (experiment tracking)

Infraestrutura

Docker

Pip / Requirements

Joblib

📦 Modelo Treinado

O arquivo model.pkl é gerado automaticamente após a execução do train.py.
Ele já inclui:

Pipeline com StandardScaler

Modelo escolhido automaticamente

Preprocessamento acoplado

📈 Fluxo da Solução (MLOps Simplificado)

│ Notebook (N2)    │
│ train.py         │
│ + MLflow Logs    │
│ model.pkl        |
│ predict.py       │
│ (inferência)     │
│ FastAPI          │
│ app.py           │
│ Docker Container │


🧩 Possíveis Extensões Futuras

Deploy em nuvem (AWS, Azure, GCP).

Monitoramento com Prometheus + Grafana.

Pipeline CI/CD automatizado (GitHub Actions).

Drift detection.

Testes unitários automáticos.

📜 Licença

Este projeto é destinado a fins educacionais.
