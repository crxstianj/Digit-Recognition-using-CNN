# Imagen de Python
FROM python:3.11

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos necesarios
COPY models.py ./models.py
COPY server.py ./server.py
COPY models/mnist_mlp.pth ./models/mnist_mlp.pth
# Instala dependencias
RUN pip install fastapi torch torchvision uvicorn pillow python-multipart

# Expone el puerto en el que correrá FastAPI
EXPOSE 3000

# Comando para ejecutar el servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]
