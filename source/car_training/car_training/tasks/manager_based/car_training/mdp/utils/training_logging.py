# car_training/utils/logging_utils.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).parents[8].resolve()


def setup_logger(
    task_name: str,
    name: str = "car_training",
    console_level: int | None = logging.INFO,
    file_level: int | None = logging.INFO,
) -> logging.Logger:
    """
    Create a dedicated logger that won't spam duplicate lines when Isaac reloads code.
    - Logs to logs/sb3/<task_name>/YYYY-MM-DD/HH-MM-SS.log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # important: don't double-print via root/Kit handlers

    # If handlers already exist, don't add again (Isaac can reload modules)
    if getattr(logger, "_configured", False):
        return logger
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S")

    log_dir = os.path.join(PROJECT_ROOT, "logs/sb3", task_name, "internal_logs", current_date)
    filename = f"{current_time}.log"

    print( f"Logging to {os.path.join(log_dir, filename)}" )


    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_fmt = logging.Formatter(
        fmt="%(levelname)s | %(message)s"
    )

    if console_level is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(console_level)
        sh.setFormatter(console_fmt)
        logger.addHandler(sh)

    if file_level is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / filename, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    logger._configured = True
    logger.info("Logger initialized: %s ", name)
    return logger


class GpuSafeLogger:
    """
    Thin wrapper around a standard logging.Logger.
    Adds lazy debug helpers so GPU-derived values are only computed when DEBUG is enabled.

    NOTE: Regular debug/info calls still evaluate args eagerly in Python.
          Use debug_lazy()/debug_tensor_* helpers to avoid unwanted CUDA sync/transfers.
    """

    def __init__(self, base: logging.Logger):
        self._base = base

    # ---- Standard logger API passthroughs ----
    @property
    def name(self) -> str:
        return self._base.name

    def setLevel(self, level: int) -> None:
        self._base.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        return self._base.isEnabledFor(level)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        # Can't prevent eager evaluation of args; this is just passthrough.
        self._base.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._base.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._base.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._base.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._base.exception(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        self._base.log(level, msg, *args, **kwargs)

    def getChild(self, suffix: str) -> "GpuSafeLogger":
        return GpuSafeLogger(self._base.getChild(suffix))

    # ---- GPU-safe helpers (lazy evaluation) ----
    def debug_lazy(self, msg: str, fn: Callable[[], Any], *args_prefix: Any, **kwargs: Any) -> None:
        """
        Evaluate fn() only if DEBUG enabled. Useful for GPU->CPU scalars or expensive formatting.
        - If fn() returns a tuple -> treated as additional format args.
        - Else -> treated as a single additional format arg.
        """
        if not self._base.isEnabledFor(logging.DEBUG):
            return

        val = fn()
        if isinstance(val, tuple):
            self._base.debug(msg, *(args_prefix + val), **kwargs)
        else:
            self._base.debug(msg, *(args_prefix + (val,)), **kwargs)

    def debug_tensor_stats(
        self,
        name: str,
        t: Any,
        every: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Debug-log mean/std/min/max of a tensor t safely (only when DEBUG enabled).
        If every/step provided, only logs when step % every == 0.
        """
        if not self._base.isEnabledFor(logging.DEBUG):
            return
        if every is not None and step is not None and (step % every != 0):
            return

        # Compute lazily inside the enabled block to avoid sync when DEBUG disabled.
        # Use .item() scalars only.
        self._base.debug(
            "%s: mean=%.4f std=%.4f min=%.4f max=%.4f shape=%s device=%s dtype=%s",
            name,
            t.mean().item(),
            t.std().item(),
            t.min().item(),
            t.max().item(),
            tuple(t.shape),
            str(t.device),
            str(t.dtype),
        )

    def __getattr__(self, item: str) -> Any:
        # Forward everything else to the base logger (handlers, addHandler, etc.)
        return getattr(self._base, item)