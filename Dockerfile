# Usa una imagen base de Python
FROM python:3.11-slim

# Instala libGL para OpenCV y otras dependencias del sistema
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Establece el directorio de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias de Python
RUN pip install -r requirements.txt

# Comando para ejecutar el bot
CMD ["python", "chatBot.py"]
