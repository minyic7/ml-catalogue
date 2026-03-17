from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
