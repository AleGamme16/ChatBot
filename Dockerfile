# Usa una imagen base de Python
FROM python:3.11-slim

# Instala libGL y otras dependencias de sistema necesarias para OpenCV y otros módulos
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias de Python
RUN pip install -r requirements.txt

# Comando para ejecutar el bot
CMD ["python", "chatBot.py"]

