from enum import Enum


class BackoffMode(Enum):
    FIBONACCI = "fibonacci"
    EXPONENTIAL = "exponential"
