FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

WORKDIR /src/llama.cpp
RUN cmake -B build \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON && \
    cmake --build build --config Release -j"$(nproc)"

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libgomp1 \
      libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY docker/railway-qwen-entrypoint.sh /usr/local/bin/railway-qwen-entrypoint.sh

RUN chmod +x /usr/local/bin/railway-qwen-entrypoint.sh

ENV MODEL_REPO=HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
    MODEL_FILE=Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
    MODEL_DIR=/tmp/models \
    N_CTX=8192 \
    N_PARALLEL=1

CMD ["railway-qwen-entrypoint.sh"]
