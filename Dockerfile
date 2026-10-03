# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Strata engine image (Qwen3.8-Flash-Next). Built and published by GitHub
# Actions from the upstream source at a pinned commit.
#
# Multi-stage: the engine is compiled on the CUDA devel base, then only the
# result, the runtime code and the Python env go into the small runtime image.
#
# Build args:
#   CUDA_ARCHITECTURES  semicolon list, e.g. "120" (RTX 50 / RTX PRO Blackwell),
#                       "89" (RTX 40), "86" (RTX 30), "75" (RTX 20).
#                       Default 120. Nvidia's CUDA images have no GPU at build
#                       time, so this must be set explicitly.
#   STRATA_REPO         upstream git URL.
#   STRATA_REF          upstream commit/tag (pinned for reproducibility).
#   CUDA_VERSION        CUDA base tag (13.0.0 by default; engine needs .so.13).
# ---------------------------------------------------------------------------
ARG CUDA_VERSION=13.0.0

# =============================== build =====================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG CUDA_ARCHITECTURES=120
ARG STRATA_REPO=https://github.com/Niko1221/Strata.git
ARG STRATA_REF=v0.1.38

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git libatomic1 libgomp1 \
        python3 python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

# upstream source at the pinned ref
WORKDIR /src
RUN git clone "$STRATA_REPO" /src/strata \
    && cd /src/strata \
    && git checkout "$STRATA_REF"

# python env (this also provides cmake + ninja used by the build)
RUN python3 -m venv /src/.venv \
    && /src/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /src/.venv/bin/pip install --no-cache-dir -r /src/strata/requirements.txt

# compile the engine exactly the way setup.py does; this also fetches llama.cpp
# at its own pinned commit.
RUN /src/.venv/bin/python - <<'PYEOF'
import os, shutil, sys
sys.path.insert(0, "/src/strata")
import setup

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
arch = os.environ["CUDA_ARCHITECTURES"].strip().strip('"').replace(",", ";")
print(f"building Strata engine for sm {arch} with {nvcc}", flush=True)

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")

eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
print("engine built:", eng / setup.EXE, flush=True)

# vision helper (llama.cpp mtmd + the mmproj projector). The encoder itself is
# chosen at run time by the config's "vision" section (GPU or CPU), so shipping
# the binary costs nothing unless the config enables it.
setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")
shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
print("vision encoder built:", eng / setup.VEXE, flush=True)
PYEOF

# ============================== runtime ====================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime
# runtime variant = CUDA runtime + math libraries (cuBLAS/cuBLASLt) + NCCL,
# which is exactly what the engine links (libcudart/libcublas/libcublasLt.so.13).

ARG STRATA_REF
ARG STRATA_VERSION
LABEL org.opencontainers.image.title="Strata" \
      org.opencontainers.image.description="Strata inference engine (Qwen3.8-Flash-Next), CUDA 13" \
      org.opencontainers.image.source="https://github.com/Niko1221/Strata" \
      org.opencontainers.image.version="${STRATA_VERSION}" \
      org.opencontainers.image.revision="${STRATA_REF}" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata

# the Python env from the build stage (same base: /usr/bin/python3.12)
COPY --from=build /src/.venv ./.venv
# runtime code, the expert routing profile, and the compiled engine
COPY --from=build /src/strata/serve ./serve
COPY --from=build /src/strata/tools ./tools
COPY --from=build /src/strata/data/expert-profile.bin ./data/expert-profile.bin
COPY --from=build /src/strata/engine/strata ./engine/strata
COPY --from=build /src/strata/engine/strata-vision ./engine/strata-vision

ENV PYTHONDONTWRITEBYTECODE=1

# /data carries the model, packs, MTP layer and the engine config (mount it).
# With setup-supported sizes you can instead point --config at a setup-generated
# config; for a manually packed model ship your own config on /data.
ENTRYPOINT ["/bin/sh", "-lc"]
CMD ["exec .venv/bin/python -m serve.server --engine strata --config /data/config/strata-ud-iq4_xs.json --port 8080"]
