# Usa uma imagem base leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANTE ---
# O comando COPY . . vai copiar o 'model.pkl' que você gerou no passo anterior
# para dentro do container, junto com os scripts .py
COPY . .

# Expõe a porta
EXPOSE 8000

# Roda a API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]