# syntax=docker/dockerfile:1
FROM python:3.11-slim

# =====================================
# CONFIGURAÇÕES BÁSICAS
# =====================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/app

# =====================================
# DEPENDÊNCIAS DO SISTEMA
# =====================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc git curl postgresql-client libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# =====================================
# COPIAR E INSTALAR DEPENDÊNCIAS PYTHON
# =====================================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================
# COPIAR RESTANTE DO PROJETO
# =====================================
COPY . .

# =====================================
# CONFIGURAÇÃO DE USUÁRIO
# =====================================
RUN chown -R appuser:appuser /home/app
USER appuser

# =====================================
# EXPOR PORTA E ENTRYPOINT
# =====================================
EXPOSE 8000

# ✅ Usa o módulo Python (evita erro de PATH)
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
