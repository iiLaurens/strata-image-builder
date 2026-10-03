# strata-image

Builds and publishes a Strata engine image to GHCR on push.

- Image: `ghcr.io/<owner>/<repo>:latest`
- Pinned upstream commit in `Dockerfile` (`STRATA_REF`); override via Actions → **publish** → Run workflow.
- `CUDA_ARCHITECTURES` defaults to `120` (RTX 50 / RTX PRO Blackwell).

```sh
docker run --rm --init --gpus all \
  --memory 64g --memory-swap 64g --memory-swappiness 0 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 30 \
  -p 127.0.0.1:8080:8080 -v /path/to/data:/data \
  ghcr.io/<owner>/<repo>:latest
```

The image expects the model volume at `/data` and runs
`.venv/bin/python -m serve.server --engine strata --config /data/config/strata-ud-iq4_xs.json --port 8080`.
