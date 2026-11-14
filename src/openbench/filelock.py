"""Small cross-process advisory file lock using only the standard library (v0.5.0).

Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows. The lock is held on a separate
lock file, so the protected data file can be read, appended to or replaced freely while it is
held. The lock is advisory: it coordinates OpenBench processes, not arbitrary programs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import TracebackType

from openbench.exceptions import OpenBenchError


class FileLockTimeoutError(OpenBenchError):
    """Raised when a file lock could not be acquired within the timeout."""


class FileLock:
    """Exclusive inter-process lock on ``path`` (created if missing).

    Re-entrant within one ``FileLock`` object, so nested ``with`` blocks in the same thread do not
    deadlock. Use one object per thread or protect it with a thread lock, as ``ResponseCache``
    does.
    """

    def __init__(self, path: str | Path, timeout_seconds: float = 30.0, poll_seconds: float = 0.02) -> None:
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._fd: int | None = None
        self._depth = 0

    def _try_lock(self, fd: int) -> bool:
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, PermissionError):
            return False
        except OSError as exc:  # pragma: no cover - unusual filesystems
            if exc.errno in (11, 35):  # EAGAIN / EWOULDBLOCK
                return False
            raise

    def _unlock(self, fd: int) -> None:
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)

    def acquire(self) -> None:
        if self._fd is not None:
            self._depth += 1
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while not self._try_lock(fd):
                if time.monotonic() >= deadline:
                    raise FileLockTimeoutError(
                        f"Could not lock {self.path} within {self.timeout_seconds:g}s; another OpenBench "
                        "process may be using the same cache. Wait for it to finish or use a different --cache-dir."
                    )
                time.sleep(self.poll_seconds)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._fd is None:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        fd, self._fd = self._fd, None
        try:
            self._unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
