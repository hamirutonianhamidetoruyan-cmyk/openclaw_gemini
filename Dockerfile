FROM node:20.11.1-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git python3 \
 && rm -rf /var/lib/apt/lists/*

COPY app/package*.json ./
RUN npm install

COPY app/ ./

RUN mkdir -p /workspace /workspace/.agent

CMD ["sh", "-c", "npm install && node index.js"]
