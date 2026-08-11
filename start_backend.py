"""
Pakistan Law Assistant — Backend launcher.
Run from the project root: python start_backend.py
"""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
from utils.config import settings

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.api_port))

    print(f"\n{'='*60}")
    print(f"  {settings.app_name} v{settings.app_version} — Backend")
    print(f"  API docs : http://localhost:{port}/docs")
    print(f"  Health   : http://localhost:{port}/api/v1/health")
    print(f"{'='*60}\n")

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=port,
        reload=False,  # disable reload in production
        log_level="info",
    )
