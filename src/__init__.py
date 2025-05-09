import logging
import sys

from groundingdino.gd_utils import config_file
from mcp_vision.server import mcp

logger = logging.getLogger(__name__)


def main():
    logger.info(f"{config_file=}")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


__all__ = ["main"]
