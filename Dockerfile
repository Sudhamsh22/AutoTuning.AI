FROM node:18-bullseye

WORKDIR /app

# ---------- install python ----------
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ---------- python deps ----------
COPY Model/requirements.txt /app/model/requirements.txt
RUN pip3 install --no-cache-dir -r /app/model/requirements.txt

# ---------- backend deps ----------
COPY Backend/package.json Backend/package-lock.json ./
RUN npm install --production

# ---------- backend source ----------
COPY Backend/ .

# ---------- frontend build ----------
WORKDIR /frontend
COPY Frontend/package.json Frontend/package-lock.json ./
RUN npm install
COPY Frontend/ .
RUN npm run build

# ---------- move frontend build ----------
WORKDIR /app
RUN mkdir -p public
RUN cp -r /frontend/dist/* ./public

# ---------- model files ----------
COPY Model ./model

ENV PORT=5000
ENV MODEL_PATH=/app/model

EXPOSE 5000

CMD ["node", "server.js"]
