from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.routes.chat import router as chat_router
from app.routes.execute import router as execute_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(execute_router)
app.include_router(chat_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        # If the path points to an actual static file, serve it directly
        static_file = (static_dir / path).resolve()
        if path and static_file.is_relative_to(static_dir) and static_file.is_file():
            return FileResponse(static_file)
        # For all other paths, serve index.html so React Router handles routing
        index = static_dir / "index.html"
        if index.is_file():
            return HTMLResponse(index.read_text())
        raise HTTPException(status_code=404)
