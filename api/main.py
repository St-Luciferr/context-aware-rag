import os
import uvicorn
from uvicorn.config import LOGGING_CONFIG
import killport
import time
import logging

from src.app import app
from src.config import settings
logger = logging.getLogger(__name__)

def is_dev():
    """Determine if we are running in development mode."""
    return settings.env.lower() in ("dev", "development", "local")


fmt = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
access_fmt='%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',

if __name__ == "__main__":
    # Kill existing dev servers only in dev mode
    if is_dev():
        killport.kill_ports(ports=[8000])
        time.sleep(1)

    # Customize logging format
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = fmt

    
    # Reload only in dev mode
    reload_flag = is_dev()
    if reload_flag:
        print("Running in dev mode")
    else:
        print("running in prod")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=reload_flag
    )
