# syntax=docker/dockerfile:1
FROM python:3.11-slim

# =====================================
# CONFIGURAÇÕES BÁSICAS
# =====================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/app/.local/bin:$PATH

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/app

# =====================================
# DEPENDÊNCIAS DO SISTEMA
# =====================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    postgresql-client \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# =====================================
# COPIAR ARQUIVOS
# =====================================
COPY . .

# =====================================
# INSTALAR DEPENDÊNCIAS PYTHON
# =====================================
# =====================================
# INSTALAR DEPENDÊNCIAS PYTHON
# =====================================
RUN pip install --upgrade pip

# ETAPA 1: Instalar dependências (isso cria o conflito)
RUN if [ -f "requirements.txt" ]; then \
        pip install -r requirements.txt; \
    else \
        echo "Aviso: Nenhum requirements.txt encontrado."; \
        pip install fastapi uvicorn pymongo python-dotenv langchain langchain-core langchain-community pydantic; \
    fi

# ETAPA 2: Corrigir o conflito de namespace do LangChain
# 1. Desinstala o 'classic' (que quebra o módulo 'agents')
# 2. Força a reinstalação do 'langchain' (o principal)
# 3. Força a reinstalação do 'langchain_agents' para RESTAURAR o namespace.
RUN pip uninstall -y langchain-classic && \
    pip install --upgrade --force-reinstall langchain

# =====================================
# CONFIGURAÇÃO DE USUÁRIO
# =====================================
RUN chown -R appuser:appuser /home/app
USER appuser

# =====================================
# EXPOR PORTA E ENTRYPOINT
# =====================================
EXPOSE 8000

# Rodar API com Uvicorn (seu entrypoint principal é api.py)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]