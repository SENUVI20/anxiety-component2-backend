from fastapi import FastAPI
from app import app as component2_app

# Vercel looks for a top-level variable named "app" in api/index.py.
# Mount the already-tested Component 2 API under /api.
app = FastAPI(title="Component 2 Vercel Gateway")
app.mount("/api", component2_app)
