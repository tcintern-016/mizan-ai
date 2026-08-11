"""
Pakistan Law Assistant — FastAPI application entry point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from utils.config import settings
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Groq model: {settings.groq_model}")
    logger.info(f"ChromaDB path: {settings.chroma_persist_dir}")
    yield
    logger.info(f"{settings.app_name} shutting down.")


app = FastAPI(
    title=settings.app_name,
    description=(
        "A RAG-based legal information assistant answering questions from "
        "Pakistani legal documents (Constitution, Penal Code, Contract Act, PECA). "
        "Educational use only — not a substitute for professional legal advice."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "ask": "/api/v1/ask (POST)",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )
