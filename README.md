# Digit Recognition using CNN

## ¿Cómo funciona?

1. La CNN se entrena sobre el dataset MNIST (60,000 imágenes de dígitos del 0 al 9)
2. El modelo entrenado se guarda en `models/mnist_mlp.pth`
3. FastAPI carga el modelo y expone un endpoint `/predict` que recibe una imagen y devuelve el dígito predicho
4. Una interfaz web permite dibujar un dígito y enviarlo directamente a la API

<p align=center>
<img width="450" src="https://github.com/user-attachments/assets/8564a89a-efbc-417c-9d69-9a1cb76556d5" />
</p>

## Arquitectura del modelo

La CNN consta de dos bloques convolucionales seguidos de un clasificador:
```
Conv2d(1→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Linear(64×7×7 → 128) → ReLU → Dropout(0.4)
Linear(128 → 10)
```

## Estructura del proyecto
```
├── main.py          # Entrenamiento del modelo
├── models.py        # Definición de CNN y MLP
├── datasets.py      # Carga del dataset MNIST
├── server.py        # API REST con FastAPI
├── pytorch_utils.py # Detección de dispositivo (CPU/GPU)
├── models/
│   └── mnist_mlp.pth  # Modelo entrenado
├── web/
│   └── index.html     # Interfaz para dibujar dígitos
└── docker-compose.yml
```

## Uso

### Con Docker (recomendado)
```bash
docker-compose up --build
```

- Interfaz web: `http://localhost:8000`
- API: `http://localhost:3000`

### Local
```bash
# Entrenar el modelo
python main.py

# Levantar la API
python server.py
```

## API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict` | Recibe una imagen y devuelve el dígito predicho |

**Ejemplo:**
```bash
curl -X POST http://localhost:3000/predict \
  -F "file=@digito.png"
```

**Respuesta:**
```json
{ "prediction": 7 }
```

## Dependencias
```bash
pip install torch torchvision fastapi uvicorn pillow scikit-learn matplotlib seaborn
```
