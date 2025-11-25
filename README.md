# 📊 Trabalho Prático: Predição de Rotatividade de Funcionários (Employee Attrition)

Este projeto apresenta uma solução completa de **Machine Learning e MLOps** para prever a probabilidade de um funcionário deixar a empresa (attrition). O sistema utiliza um modelo de **Regressão Logística** treinado com dados históricos e disponibilizado através de uma **API REST** containerizada com **Docker**.

## 👨‍🎓 Alunos
* **Diego Planinscheick**
* **Gabriel Henrique Karing**
* **José Marcos Zoz Marques**

---

## 📂 Estrutura do Projeto

A organização dos arquivos segue o padrão abaixo:

```text
/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset (IBM/Kaggle)
├── app.py              # Aplicação principal (API FastAPI)
├── Dockerfile          # Receita para criar a imagem Docker
├── predict.py          # Módulo de inferência (carrega o modelo)
├── train.py            # Script de treinamento e geração do modelo
├── requirements.txt    # Lista de bibliotecas Python
├── model.pkl           # Artefato do modelo (gerado após o treino)
└── README.md           # Documentação
🛠️ Pré-requisitos
Para executar este projeto localmente, você precisará de:

Python 3.9 ou superior.

Docker Desktop instalado e em execução.

🚀 Passo a Passo para Execução
Siga a ordem abaixo para garantir que o modelo seja treinado antes de subir a aplicação.

1. Instalação das Dependências
No terminal, dentro da pasta do projeto, instale as bibliotecas necessárias:

Bash

pip install -r requirements.txt
2. Treinamento do Modelo (Essencial)
Antes de criar o container, é necessário gerar o arquivo do modelo (model.pkl). Execute o script de treino:

Bash

python train.py
O que isso faz? Processa os dados, treina o modelo, exibe as métricas (Acurácia, Recall, AUC) e salva o arquivo model.pkl na raiz do projeto.

3. Construção da Imagem Docker
Com o modelo gerado, construa a imagem Docker. Isso vai empacotar o código e o modelo em um ambiente isolado:

Bash

docker build -t hr-attrition-api .
4. Executando a API
Inicie o container mapeando a porta 8000:

Bash

docker run -p 8000:8000 hr-attrition-api
A API estará rodando em: http://localhost:8000

🧪 Como Testar a API
Via Swagger UI (Visual)
O FastAPI fornece uma documentação interativa automática.

Acesse: http://localhost:8000/docs

Clique em POST /predict.

Clique no botão Try it out.

Insira o JSON de exemplo abaixo e clique em Execute.

Exemplo de JSON para Teste
Você pode usar este corpo para testar a requisição:

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