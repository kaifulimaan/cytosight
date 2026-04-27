---
title: CytoSight
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# CytoSight API

Medical classification and segmentation platform backend.

## Deployment Info
This Space runs a FastAPI server inside a Docker container.
- **Port:** 7860
- **Framework:** FastAPI
- **Model Backbone:** Phikon v2

## Setup
Ensure the following Secrets are set in the Space settings:
- `HF_TOKEN`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `OPENAI_API_KEY`
