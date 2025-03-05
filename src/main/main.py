from loguru import logger

from ..log.config import configure_logging


def main():
    configure_logging()

    logger.info("Hello, world!")


if __name__ == "__main__":
    main()
