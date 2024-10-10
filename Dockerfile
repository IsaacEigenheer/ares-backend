FROM python:3.10-slim

# Instala as dependências do sistema e o 
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO- https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs=18.19.1-1nodesource1 \
    && npm install -g npm@10.2.4

RUN pip install --upgrade pip

# Copia e instala as dependências do Python
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Instala o Google Chrome
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    apt-get clean

# Define o diretório de trabalho
WORKDIR /

# Copia o código da aplicação para o contêiner
COPY . .

# Instala as dependências do Node.js
RUN npm install --quiet --no-optional --no-fund --loglevel=error

# Expõe a porta para acesso
EXPOSE 7001

# Comando para iniciar a aplicação
CMD ["sh", "-c", "sleep 5 && npm run start:dev"]

