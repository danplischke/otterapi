"""Undo log for the files one generation run writes.

``Codegen.generate()`` has to leave the output directory as it found it when a
run fails partway.  The first implementation snapshotted every file under the
directory and, on failure, deleted the lot and wrote the snapshot back.  That
read whatever lived beside the generated code (a virtualenv, when the output is
``.``), gave up at the first entry it could not delete -- a symlinked directory,
a FIFO -- after the deleting and before the writing back, which lost every file,
and even when it finished it rewrote files with default permissions, turned
symlinks into copies and dropped empty directories.

``WriteJournal`` records instead what generation itself touches, just before it
touches it, so there is nothing to undo that the run did not do.  Every write
site goes through the helpers at the bottom of this module.  They use the
journal that :func:`activate` made current and fall back to plain filesystem
calls when there is none, so a builder run on its own (as the tests do) still
writes.

Because undoing means deleting, the journal is deliberately conservative about
it:

* It never deletes recursively.  A created file is unlinked, a created
  directory is ``rmdir``-ed, and a directory that is not empty (something the
  run did not record put a file in it) is left in place with a warning.  A
  bookkeeping mistake can therefore leave an empty directory behind; it cannot
  empty a directory that holds data.
* It stays inside the output directory it was built for.  Recording a path
  outside it is an error, and so is a symlink inside it that leads outside, so a
  generated file cannot be redirected onto somebody else's file.  The
  containment is checked again at the moment of each undo, not only when the
  entry was recorded.
* It never follows a symlink to delete: ``unlink`` removes a link, not its
  target, and ``rmdir`` refuses one.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from upath import UPath

from otterapi.exceptions import OutputError

__all__ = [
    'WriteJournal',
    'activate',
    'ensure_dir',
    'record_file',
    'touch',
    'write_text',
]

logger = logging.getLogger(__name__)

AnyPath = Path | UPath


def _is_local(path: AnyPath) -> bool:
    """Whether *path* is on the local filesystem, where symlinks and modes exist."""
    return isinstance(path, Path)


def _contains(root: AnyPath, path: AnyPath, *, allow_root: bool) -> bool:
    """Whether *path* is *root* (if ``allow_root``) or strictly inside it.

    Local paths are compared after resolving symlinks, so a link inside the
    root that points out of it does not count as inside.  Other filesystems
    have no links, and their paths are compared by components.
    """
    if _is_local(root) and _is_local(path):
        real_root = Path(os.path.realpath(str(root)))
        real_path = Path(os.path.realpath(str(path)))
        if real_path == real_root:
            return allow_root
        return real_root in real_path.parents
    if path == root:
        return allow_root
    return root in path.parents


def _require_inside(root: AnyPath, path: AnyPath, *, allow_root: bool = False) -> None:
    """Raise unless *path* is inside *root*; see :func:`_contains`."""
    if not _contains(root, path, allow_root=allow_root):
        raise OutputError(
            str(path),
            cause=ValueError(f'it is outside the output directory {root}'),
        )


def _write_target(path: AnyPath) -> AnyPath:
    """The file a write to *path* really lands on.

    A write goes through a symlink, so the file at stake is its target -- which
    a dangling link does not have yet.  Journalled by the link itself, rollback
    would delete the user's link and leave the file the run created behind it.
    """
    if _is_local(path) and path.is_symlink():
        return Path(os.path.realpath(str(path)))
    return path


@dataclass(frozen=True)
class _CreatedFile:
    """A file the run created; rolling back unlinks it."""

    path: AnyPath

    def undo(self, root: AnyPath) -> None:
        _require_inside(root, self.path)
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _OverwrittenFile:
    """A file the run overwrote; rolling back writes its old contents back."""

    path: AnyPath
    content: bytes
    #: Permission bits, recorded for a local path only: a remote filesystem has
    #: none to restore.
    mode: int | None

    def undo(self, root: AnyPath) -> None:
        _require_inside(root, self.path)
        self.path.write_bytes(self.content)
        # Only when they differ: chmod needs ownership where a write needs just
        # the group bit, so a shared directory would warn for nothing.
        if (
            self.mode is not None
            and stat.S_IMODE(self.path.stat().st_mode) != self.mode
        ):
            self.path.chmod(self.mode)


@dataclass(frozen=True)
class _CreatedDir:
    """A directory the run created; rolling back ``rmdir``-s it if it is empty."""

    path: AnyPath

    def undo(self, root: AnyPath) -> None:
        _require_inside(root, self.path, allow_root=True)
        # Not recursive, on purpose: see the module docstring.
        self.path.rmdir()


_Entry = _CreatedFile | _OverwrittenFile | _CreatedDir


class WriteJournal:
    """Records what one run creates or overwrites so it can be undone.

    The journal never looks at anything the run does not write, so its cost
    follows the size of the generated package, not of whatever else sits in the
    output directory.

    Args:
        root: The output directory.  Every path the journal records or undoes
            must be inside it.

    Raises:
        OutputError: If *root* is the filesystem root.
    """

    def __init__(self, root: AnyPath) -> None:
        if root.parent == root:
            raise OutputError(
                str(root), cause=ValueError('the output directory cannot be the root')
            )
        self._root = root
        self._entries: list[_Entry] = []
        self._recorded: set[AnyPath] = set()

    def record_file(self, path: AnyPath) -> None:
        """Note what is at *path*; call it before creating or overwriting the file.

        Only the first call for a path counts, since that is the state to
        return to.  An existing file is read into memory (with its permission
        bits, for a local path); a missing one is noted as created by this run.

        Raises:
            OutputError: If *path*, or the file a symlink at *path* leads to, is
                outside the output directory.
            OSError: If an existing file cannot be read.  Generation then fails
                before it writes anything, instead of overwriting a file it
                could not put back.
        """
        target = _write_target(path)
        _require_inside(self._root, target)
        if target in self._recorded:
            return

        if target.is_file():
            mode = stat.S_IMODE(target.stat().st_mode) if _is_local(target) else None
            self._entries.append(_OverwrittenFile(target, target.read_bytes(), mode))
        elif not target.exists():
            self._entries.append(_CreatedFile(target))
        # Otherwise a directory, FIFO or the like is in the way: the write fails
        # and there is nothing of the run's to undo.
        self._recorded.add(target)

    def ensure_dir(self, path: AnyPath) -> None:
        """Create *path* and any missing parents, noting each directory it makes.

        Equivalent to ``path.mkdir(parents=True, exist_ok=True)``, except that
        the parents are made one at a time so each can be recorded, and only
        the ones that did not exist before.  Missing parents *above* the output
        directory are created but not recorded: rolling back never removes
        anything outside it.

        Raises:
            OutputError: If *path* is outside the output directory.
        """
        _require_inside(self._root, path, allow_root=True)

        missing: list[AnyPath] = []
        while not path.is_dir():
            missing.append(path)
            if path.parent == path:
                break
            path = path.parent

        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                # Somebody else made it since the check above, which makes it
                # theirs to clean up.  Anything but a directory is a conflict.
                if not directory.is_dir():
                    raise
            else:
                if _contains(self._root, directory, allow_root=True):
                    self._entries.append(_CreatedDir(directory))

    def rollback(self) -> None:
        """Undo everything recorded, newest first.

        Best-effort and never raises: it runs while the error that caused it is
        propagating, so a failure is logged and the remaining entries are still
        undone.  A directory comes off after the files in it, because it was
        made first, and only if it is then empty.
        """
        while self._entries:
            entry = self._entries.pop()
            try:
                entry.undo(self._root)
            except Exception as exc:
                logger.warning(
                    'Could not roll back %s after a failed generation: %s',
                    entry.path,
                    exc,
                )
        self._recorded.clear()


_active: ContextVar[WriteJournal | None] = ContextVar(
    'otterapi_write_journal', default=None
)


@contextmanager
def activate(journal: WriteJournal) -> Iterator[None]:
    """Make *journal* the one the module-level helpers record into."""
    token = _active.set(journal)
    try:
        yield
    finally:
        _active.reset(token)


def record_file(path: AnyPath) -> None:
    """Record *path* in the active journal, if any, before it is written.

    The helpers below call this themselves; it is for a write that goes through
    some other API.
    """
    journal = _active.get()
    if journal is not None:
        journal.record_file(path)


def ensure_dir(path: AnyPath) -> None:
    """``path.mkdir(parents=True, exist_ok=True)``, journalled when active."""
    journal = _active.get()
    if journal is None:
        path.mkdir(parents=True, exist_ok=True)
    else:
        journal.ensure_dir(path)


def write_text(path: AnyPath, content: str, encoding: str = 'utf-8') -> None:
    """Write *content* to *path*, journalled when active.

    The encoding defaults to UTF-8 rather than the platform locale: generated
    modules are UTF-8 source, and the locale broke the export module on Windows.
    """
    record_file(path)
    path.write_text(content, encoding=encoding)


def touch(path: AnyPath) -> None:
    """Create *path* if it is missing, journalled when active."""
    record_file(path)
    path.touch()
