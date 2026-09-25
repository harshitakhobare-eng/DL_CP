"""Application Launcher for MathTutor AI.

Starts the FastAPI server with Uvicorn on configured host and port.
"""

import os
import uvicorn
import yaml

if __name__ == "__main__":
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    ui_cfg = config.get("ui", {})
    host = ui_cfg.get("host", "127.0.0.1")
    port = ui_cfg.get("port", 8000)

    print(f"\n" + "=" * 60)
    print(f" Starting MathTutor AI Application...")
    print(f" Web Interface: http://{host}:{port}")
    print(f" API Docs:      http://{host}:{port}/docs")
    print(f"=" * 60 + "\n")

    uvicorn.run("app.api.server:app", host=host, port=port, reload=False)
