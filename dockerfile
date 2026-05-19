FROM python:3.11-slim

# Requerido por Hugging Face para permisos correctos
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copiar e instalar dependencias primero (aprovecha la caché de Docker)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copiar el resto del código y el modelo
COPY --chown=user . /app

# Nota: Tu archivo de código debe llamarse obligatoriamente "app.py"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]