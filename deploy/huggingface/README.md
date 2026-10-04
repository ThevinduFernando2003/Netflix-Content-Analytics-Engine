# Hugging Face Spaces (Docker SDK)

1. Create a Space, SDK = Docker.
2. Point it at this repository. The root `Dockerfile` is the runtime image.
3. Bake `registry/`, `artifacts/` and `data/features/` into the image **before** push — do not train at startup.
4. Set `PORT` (Spaces provides it). Set `API_BASE_URL` if the dashboard is a second Space.
5. No secrets are required. Leave `ENRICHMENT_ENABLED=false`.

A single Space can serve the API; a second Space can serve Streamlit with `API_BASE_URL` pointing at the first. Free-tier idle eviction does not apply to Spaces the way it does on Render.
