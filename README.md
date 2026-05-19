# Clasificación de Dígitos Manuscritos con MNIST

API en Python para clasificar imágenes manuscritas de dígitos (0–9) usando un modelo CNN entrenado con el dataset MNIST.

**Repositorios:**
- GitHub: https://github.com/jeilopez1/PruebaTec_BdB
- Hugging Face Space: https://huggingface.co/spaces/jeilopez/PruebaTec_BdB
- API Docs (Swagger): https://jeilopez-pruebatec-bdb.hf.space/docs

## Requisitos

- Python 3.11+
- Dependencias para la API: `requirements.txt` (raíz)
- Dependencias para el notebook: `Entrenamiento/requirements.txt`

## Cómo ejecutar el notebook

```bash
# Crear y activar entorno virtual (opcional)
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias del notebook
pip install -r Entrenamiento/requirements.txt

# Abrir el notebook
jupyter notebook Entrenamiento/MainEntrenamiento.ipynb
```

O desde VS Code, abrir `Entrenamiento/MainEntrenamiento.ipynb` y ejecutar todas las celdas.

El notebook realiza:

- Carga del dataset MNIST desde Keras
- Análisis exploratorio y balance de clases
- Preprocesamiento (normalización [0,1], reshape para CNN)
- Comparación controlada entre Regresión Logística, MLP y CNN
- Evaluación con curvas de precisión/pérdida, matriz de confusión y métricas
- Validación visual con 32 ejemplos (etiqueta real vs. predicha)
- Exportación del modelo seleccionado

## Cómo exportar el modelo

El notebook exporta automáticamente el mejor modelo CNN al archivo `modelo_mnist_final.keras` en la raíz del proyecto.

Para reexportar manualmente desde el notebook, ejecutar la celda correspondiente a la exportación (última celda del notebook).

## Cómo levantar la API

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en `http://localhost:8000`.

### Docker

```bash
docker build -t mnist-api .
docker run -p 7860:7860 mnist-api
```

### Hugging Face Space (ya desplegado)

https://jeilopez-pruebatec-bdb.hf.space/docs

### Endpoints

| Método | Ruta         | Descripción                     |
|--------|--------------|---------------------------------|
| GET    | `/`          | Información general             |
| GET    | `/health`    | Estado del servicio y modelo    |
| POST   | `/predict`   | Predecir dígito desde imagen    |

Documentación interactiva disponible en `/docs` (Swagger UI).

## Ejemplo de consumo del endpoint

### Python (requests)

```python
import requests
import base64

with open("imagen.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post("https://jeilopez-pruebatec-bdb.hf.space/predict", json={
    "image_base64": b64
})

print(resp.json())  # {"prediction": 7}
```

### cURL

```bash
curl -X POST https://jeilopez-pruebatec-bdb.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'"$(base64 -w0 imagen.png)"'"}'
```

### Respuesta esperada

```json
{"prediction": 7}
```

### Manejo de errores

La API retorna códigos HTTP adecuados:

| Código | Descripción                          |
|--------|--------------------------------------|
| 400    | base64 inválido, imagen vacía o formato no válido |
| 500    | Error interno del servidor           |

Ejemplo de error:

```json
{"detail": "base64 inválido"}
```

## Estructura del proyecto

```
├── app.py                          # API FastAPI
├── modelo_mnist_final.keras        # Modelo CNN exportado
├── requirements.txt                # Dependencias API
├── preprocessing_info.json         # Metadatos del preprocesamiento
├── Dockerfile                      # Imagen Docker
├── .gitignore
└── Entrenamiento/
    ├── MainEntrenamiento.ipynb     # Notebook de entrenamiento
    └── requirements.txt            # Dependencias del notebook
```
