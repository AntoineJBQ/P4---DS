# Utilisez une image de base appropriée
FROM python:3.9-slim

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers nécessaires dans le conteneur
COPY . .

# Installez les dépendances de votre application
RUN pip install --no-cache-dir -r requirements.txt

# Exécutez votre application lorsque le conteneur démarre
CMD ["python", "app.py"]
