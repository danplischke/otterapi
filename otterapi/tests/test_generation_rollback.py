"""Tests that a failed generation undoes what it did, and only that.

Generation used to roll back by snapshotting every file under the output
directory and, on failure, deleting the lot and writing the snapshot back. That
read files it had no business reading (a virtualenv, for ``-o .``), lost the
whole directory when one entry could not be deleted -- a symlinked directory, a
FIFO -- and, when it did finish, rewrote files with default permissions, turned
symlinks into copies and dropped empty directories. A spec author could force
the rollback at will: two ``servers`` entries fail after ``models.py`` is
written.

The rollback is now a write journal (``otterapi.codegen._write_journal``) that
records only the files and directories the run itself creates or overwrites.
These tests pin what follows from that: everything else is left exactly as it
was, what the run did write is put back exactly, and no file the run does not
write is even read.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import stat
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
from upath import UPath

from otterapi.codegen._features import PaginationFeature
from otterapi.codegen._write_journal import (
    WriteJournal,
    _CreatedDir,
    _CreatedFile,
    _OverwrittenFile,
    activate,
    ensure_dir,
    record_file,
    touch,
    write_text,
)
from otterapi.codegen.codegen import Codegen
from otterapi.codegen.export import generate_export_module
from otterapi.codegen.pagination import generate_pagination_module
from otterapi.codegen.utils import write_mod
from otterapi.config import (
    DataFrameConfig,
    DocumentConfig,
    ExportConfig,
    ModuleSplitConfig,
    PaginationConfig,
)
from otterapi.exceptions import OutputError
from otterapi.tests.test_atomic_generation import (
    MULTI_SERVER_SPEC,
    VALID_SPEC,
    _generate,
    _write_spec,
)

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == 'win32',
    reason='needs POSIX permission bits, symlinks or FIFOs',
)

JOURNAL_LOGGER = 'otterapi.codegen._write_journal'

# Generation gets as far as writing models.py, then fails on the second server.
# Unlike MULTI_SERVER_SPEC the schema differs from VALID_SPEC's, so the models.py
# it writes is not byte-identical to the one a previous good run left behind.
CHANGED_MULTI_SERVER_SPEC = {
    **MULTI_SERVER_SPEC,
    'components': {
        'schemas': {
            'Thing': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'size': {'type': 'integer'},
                },
            }
        }
    },
}

# The same change, with a single server: generation succeeds.
CHANGED_VALID_SPEC = {
    **CHANGED_MULTI_SERVER_SPEC,
    'servers': VALID_SPEC['servers'],
}

# Two operations, so a module map can put them in different modules and, nested,
# in different directories.
LAYOUT_SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'Layout', 'version': '1.0'},
    'servers': [{'url': 'https://example.test'}],
    'paths': {
        '/pet/{petId}': {
            'get': {
                'operationId': 'getPet',
                'parameters': [
                    {
                        'name': 'petId',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'integer'},
                    }
                ],
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Pet'}
                            }
                        },
                    }
                },
            }
        },
        '/store/order': {
            'get': {
                'operationId': 'getOrder',
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {'application/json': {'schema': {'type': 'object'}}},
                    }
                },
            }
        },
    },
    'components': {
        'schemas': {
            'Pet': {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        }
    },
}


def _layout_config(layout: str) -> dict:
    """Config kwargs for one of the three file layouts, with every feature on.

    Every optional runtime module is switched on so each ``FeatureModule`` write
    is exercised, and the nested layout is three levels deep so that the
    intermediate ``api/__init__.py`` is the empty, touched kind.
    """
    config: dict = {
        'pagination': PaginationConfig(enabled=True),
        'export': ExportConfig(enabled=True),
        'dataframe': DataFrameConfig(enabled=True),
    }
    modules = {'pets': ['/pet/**'], 'orders': ['/store/**']}
    if layout == 'split-flat':
        config['module_split'] = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            min_endpoints=1,
            flat_structure=True,
            module_map=modules,
        )
    elif layout == 'split-nested':
        config['module_split'] = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            min_endpoints=1,
            module_map={'api': {'store': modules}},
        )
    return config


class _FailsAtTheEnd(Codegen):
    """Writes the whole package, then fails: the worst case for a rollback."""

    failure: type[BaseException] = RuntimeError

    def _generate_into(self, directory):
        super()._generate_into(directory)
        raise self.failure('late failure')


def _generate_then_fail(
    spec_file: Path, output: Path, failure: type[BaseException] = RuntimeError, **kwargs
) -> None:
    """Generate every file, then fail, and check that it was the late failure."""

    class Failing(_FailsAtTheEnd):
        pass

    Failing.failure = failure
    codegen = Failing(
        DocumentConfig(source=str(spec_file), output=str(output), **kwargs)
    )
    with pytest.raises(failure, match='late failure'):
        codegen.generate()


def _write_named_spec(tmp_path: Path, name: str, spec: dict) -> Path:
    spec_file = tmp_path / name
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    return spec_file


def _fail_early(spec_file: Path, output: Path, **kwargs) -> None:
    """Fail on the second server, after only models.py has been written."""
    with pytest.raises(ValueError, match='Multiple servers'):
        _generate(spec_file, output, **kwargs)


def _tree(root: Path) -> dict[str, tuple]:
    """Everything under *root* -- kind, permissions, contents -- unfollowed.

    Symlinks are recorded as their target text and never entered, so two equal
    trees are equal entry for entry, which is what "left as it was" means.
    """
    entries: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath, name)
            info = path.lstat()
            key = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                entries[key] = ('symlink', os.readlink(path))
            elif stat.S_ISREG(info.st_mode):
                entries[key] = ('file', stat.S_IMODE(info.st_mode), path.read_bytes())
            elif stat.S_ISDIR(info.st_mode):
                entries[key] = ('dir', stat.S_IMODE(info.st_mode))
            else:
                entries[key] = ('special', stat.S_IFMT(info.st_mode))
    return entries


@contextmanager
def _watching_opens() -> Iterator[list[str]]:
    """Collect the real path of every file opened, however it was opened.

    A bare ``open()`` goes through ``builtins.open`` and ``io.open``; pathlib's
    ``read_bytes`` and ``open`` go through ``Path.open``. Before Python 3.12
    pathlib bound ``io.open`` when it was imported, so patching ``io.open`` alone
    sees none of its reads there -- ``Path.open`` has to be patched as well.
    """
    opened: list[str] = []
    real_open = io.open
    real_path_open = Path.open

    def spy(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            opened.append(os.path.realpath(os.fsdecode(file)))
        return real_open(file, *args, **kwargs)

    def path_spy(self, *args, **kwargs):
        opened.append(os.path.realpath(str(self)))
        return real_path_open(self, *args, **kwargs)

    with (
        mock.patch.object(io, 'open', spy),
        mock.patch.object(builtins, 'open', spy),
        mock.patch.object(Path, 'open', path_spy),
    ):
        yield opened


def _directory_with_awkward_entries(output: Path) -> None:
    """Fill *output* with what the old rollback could not remove.

    This is roughly ``-o .`` in a project: a virtualenv with its ``lib64 ->
    lib`` symlink, a FIFO, a dangling link, an empty directory, a private one.
    """
    (output / 'lib').mkdir(parents=True)
    (output / 'lib' / 'site.py').write_text('print("hi")\n', encoding='utf-8')
    (output / 'lib64').symlink_to('lib', target_is_directory=True)
    os.mkfifo(output / 'events.pipe')
    (output / 'dangling').symlink_to('does-not-exist')
    (output / 'empty').mkdir()
    (output / 'private').mkdir()
    (output / 'private' / 'key.pem').write_text('-----BEGIN-----\n', encoding='utf-8')
    (output / 'private').chmod(0o700)
    (output / 'notes.txt').write_text('keep me', encoding='utf-8')


class TestUnrelatedEntriesAreLeftAlone:
    @POSIX_ONLY
    def test_files_keep_their_bytes_modes_and_mtimes(self, tmp_path: Path):
        output = tmp_path / 'client'
        output.mkdir()
        secret = output / 'secrets.env'
        secret.write_bytes(b'TOKEN=abc\n')
        secret.chmod(0o600)
        script = output / 'run.sh'
        script.write_bytes(b'#!/bin/sh\necho hi\n')
        script.chmod(0o755)
        before = {p: (p.read_bytes(), p.stat()) for p in (secret, script)}

        _fail_early(_write_spec(tmp_path, MULTI_SERVER_SPEC), output)

        for path, (content, info) in before.items():
            now = path.stat()
            assert path.read_bytes() == content
            assert stat.S_IMODE(now.st_mode) == stat.S_IMODE(info.st_mode)
            # Not even rewritten with identical bytes.
            assert now.st_mtime_ns == info.st_mtime_ns
        assert stat.S_IMODE(secret.stat().st_mode) == 0o600
        assert os.access(script, os.X_OK)

    @POSIX_ONLY
    def test_a_directory_symlink_a_fifo_and_empty_directories_survive(
        self, tmp_path: Path
    ):
        """The case that used to destroy everything.

        The old restore unlinked files, then hit ``lib64`` -- a symlink that
        ``is_dir()`` -- and died in ``rmdir()`` before writing the snapshot
        back, so ``site.py`` and ``notes.txt`` were simply gone.
        """
        output = tmp_path / 'client'
        output.mkdir()
        _directory_with_awkward_entries(output)
        before = _tree(output)

        _fail_early(_write_spec(tmp_path, MULTI_SERVER_SPEC), output)

        assert (output / 'lib64').is_symlink()
        assert os.readlink(output / 'lib64') == 'lib'
        assert stat.S_ISFIFO((output / 'events.pipe').lstat().st_mode)
        assert (output / 'dangling').is_symlink()
        assert (output / 'empty').is_dir()
        assert (output / 'lib' / 'site.py').read_text(encoding='utf-8') == (
            'print("hi")\n'
        )
        assert (output / 'notes.txt').read_text(encoding='utf-8') == 'keep me'
        # ...and nothing else was deleted or left behind (models.py is gone).
        assert _tree(output) == before

    @POSIX_ONLY
    def test_symlinks_out_of_the_directory_stay_symlinks(self, tmp_path: Path):
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'secret.txt').write_text('keep', encoding='utf-8')
        output = tmp_path / 'client'
        output.mkdir()
        (output / 'linked-dir').symlink_to(outside, target_is_directory=True)
        (output / 'linked-file').symlink_to(outside / 'secret.txt')
        before_output, before_outside = _tree(output), _tree(outside)

        _fail_early(_write_spec(tmp_path, MULTI_SERVER_SPEC), output)

        assert (output / 'linked-dir').is_symlink()
        assert Path(os.readlink(output / 'linked-dir')) == outside
        assert (output / 'linked-file').is_symlink()
        assert Path(os.readlink(output / 'linked-file')) == outside / 'secret.txt'
        assert _tree(output) == before_output
        assert _tree(outside) == before_outside


class TestWhatTheRunWroteIsPutBack:
    def test_an_overwritten_models_file_comes_back_byte_for_byte(self, tmp_path: Path):
        output = tmp_path / 'client'
        _generate(_write_spec(tmp_path, VALID_SPEC), output)
        models = output / 'models.py'
        if sys.platform != 'win32':
            models.chmod(0o640)
        original = models.read_bytes()
        before = _tree(output)

        # The failing run writes a different models.py (checked on a control run
        # so the comparison below is not between two identical files).
        _generate(
            _write_named_spec(tmp_path, 'changed.json', CHANGED_VALID_SPEC),
            tmp_path / 'control',
        )
        assert (tmp_path / 'control' / 'models.py').read_bytes() != original

        _fail_early(
            _write_named_spec(tmp_path, 'bad.json', CHANGED_MULTI_SERVER_SPEC), output
        )

        assert models.read_bytes() == original
        if sys.platform != 'win32':
            assert stat.S_IMODE(models.stat().st_mode) == 0o640
        assert _tree(output) == before

    def test_a_previous_client_is_restored_after_every_file_was_rewritten(
        self, tmp_path: Path
    ):
        """The run got to the very end, so every generated file was touched."""
        output = tmp_path / 'client'
        _generate(_write_spec(tmp_path, VALID_SPEC), output)
        before = _tree(output)
        changed = _write_named_spec(tmp_path, 'changed.json', CHANGED_VALID_SPEC)
        _generate(changed, tmp_path / 'control')
        assert _tree(tmp_path / 'control') != before

        _generate_then_fail(changed, output)

        assert _tree(output) == before


class TestDirectoriesTheRunCreated:
    def test_a_missing_output_directory_goes_and_the_parents_made_for_it_stay(
        self, tmp_path: Path
    ):
        """The journal never removes anything outside the output directory."""
        output = tmp_path / 'a' / 'b' / 'client'

        _fail_early(_write_spec(tmp_path, MULTI_SERVER_SPEC), output)

        assert not output.exists()
        assert (tmp_path / 'a' / 'b').is_dir()
        assert list((tmp_path / 'a' / 'b').iterdir()) == []

    def test_an_existing_output_directory_loses_exactly_the_subdirectories_it_gained(
        self, tmp_path: Path
    ):
        spec_file = _write_spec(tmp_path, LAYOUT_SPEC)
        output = tmp_path / 'client'
        (output / 'kept' / 'inner').mkdir(parents=True)
        (output / 'kept' / 'inner' / 'data.txt').write_text('x', encoding='utf-8')
        (output / 'empty').mkdir()
        (output / 'notes.txt').write_text('keep me', encoding='utf-8')
        before = _tree(output)

        _generate_then_fail(spec_file, output, **_layout_config('split-nested'))

        assert not (output / 'api').exists()
        assert _tree(output) == before

    def test_the_output_directory_is_removed_when_nothing_was_there(
        self, tmp_path: Path
    ):
        spec_file = _write_spec(tmp_path, LAYOUT_SPEC)
        output = tmp_path / 'client'

        _generate_then_fail(spec_file, output, **_layout_config('split-nested'))

        assert not output.exists()

    def test_an_interrupt_rolls_back_too(self, tmp_path: Path):
        spec_file = _write_spec(tmp_path, LAYOUT_SPEC)
        output = tmp_path / 'client'

        _generate_then_fail(spec_file, output, failure=KeyboardInterrupt)

        assert not output.exists()


class TestEveryWriteIsJournalled:
    """Each write site must go through the journal, or it escapes the rollback.

    The output directory is pre-populated with the (empty) directories the run
    would otherwise create, so a file written behind the journal's back stays
    behind, and the comparison with the tree before the run fails.
    """

    @pytest.mark.parametrize('layout', ['flat', 'split-flat', 'split-nested'])
    def test_a_late_failure_leaves_nothing_behind(self, tmp_path: Path, layout: str):
        spec_file = _write_spec(tmp_path, LAYOUT_SPEC)
        config = _layout_config(layout)

        scratch = tmp_path / 'scratch'
        _generate(spec_file, scratch, **config)
        output = tmp_path / 'client'
        output.mkdir()
        for directory in sorted(scratch.rglob('*'), key=lambda p: len(p.parts)):
            if directory.is_dir():
                (output / directory.relative_to(scratch)).mkdir(exist_ok=True)
        before = _tree(output)
        assert {path.name for path in scratch.rglob('*') if path.is_file()} >= {
            'models.py',
            '_client.py',
            'client.py',
            '__init__.py',
            'py.typed',
            '_pagination.py',
            '_export.py',
            '_dataframe.py',
            '_concurrency.py',
            '_retry.py',
            '_serialization.py',
        }

        _generate_then_fail(spec_file, output, **config)

        assert _tree(output) == before

    def test_the_nested_layout_includes_a_touched_package_marker(self, tmp_path: Path):
        """Guards the previous test: ``touch`` is what writes ``api/__init__.py``."""
        scratch = tmp_path / 'scratch'
        _generate(
            _write_spec(tmp_path, LAYOUT_SPEC),
            scratch,
            **_layout_config('split-nested'),
        )

        assert (scratch / 'api' / '__init__.py').read_bytes() == b''
        assert (scratch / 'api' / 'store' / 'pets.py').exists()


class TestUnrelatedFilesAreNotRead:
    """The snapshot read the whole tree; the journal reads only what it overwrites."""

    @pytest.mark.parametrize('outcome', ['fails', 'succeeds'])
    def test_no_file_the_run_does_not_write_is_opened(
        self, tmp_path: Path, outcome: str
    ):
        output = tmp_path / 'client'
        (output / 'sub').mkdir(parents=True)
        unrelated = [
            output / 'huge.bin',
            output / 'notes.txt',
            output / 'sub' / 'data.txt',
        ]
        for path in unrelated:
            path.write_bytes(b'unrelated')
        # An older client.py is the user's, and a previous models.py is one the
        # run overwrites: the control that shows the spy would see a read.
        (output / 'models.py').write_text('# old\n', encoding='utf-8')
        spec = MULTI_SERVER_SPEC if outcome == 'fails' else VALID_SPEC
        spec_file = _write_spec(tmp_path, spec)

        with _watching_opens() as opened:
            if outcome == 'fails':
                _fail_early(spec_file, output)
            else:
                _generate(spec_file, output)

        assert os.path.realpath(output / 'models.py') in opened
        for path in unrelated:
            assert os.path.realpath(path) not in opened

    def test_an_unreadable_file_fails_the_run_before_it_writes_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An existing file that cannot be read cannot be put back, so leave it."""
        output = tmp_path / 'client'
        output.mkdir()
        models = output / 'models.py'
        models.write_text('# precious\n', encoding='utf-8')
        real_read_bytes = Path.read_bytes

        def read_bytes(self: Path) -> bytes:
            if self.name == 'models.py':
                raise PermissionError(13, 'Permission denied', str(self))
            return real_read_bytes(self)

        monkeypatch.setattr(Path, 'read_bytes', read_bytes)

        with pytest.raises(PermissionError):
            _generate(_write_spec(tmp_path, VALID_SPEC), output)

        assert models.read_text(encoding='utf-8') == '# precious\n'
        assert [p.name for p in output.iterdir()] == ['models.py']


class TestWriteJournal:
    def test_the_first_pre_image_wins(self, tmp_path: Path):
        target = tmp_path / 'models.py'
        target.write_text('original', encoding='utf-8')
        journal = WriteJournal(tmp_path)

        journal.record_file(target)
        target.write_text('first write', encoding='utf-8')
        journal.record_file(target)  # would capture 'first write' if it counted
        target.write_text('second write', encoding='utf-8')
        journal.rollback()

        assert target.read_text(encoding='utf-8') == 'original'

    def test_created_files_are_deleted_and_overwritten_ones_restored(
        self, tmp_path: Path
    ):
        old = tmp_path / 'old.py'
        old.write_bytes(b'old\r\n\x00not text')
        new = tmp_path / 'new.py'
        journal = WriteJournal(tmp_path)

        journal.record_file(old)
        old.write_text('overwritten', encoding='utf-8')
        journal.record_file(new)
        new.write_text('created', encoding='utf-8')
        journal.rollback()

        assert old.read_bytes() == b'old\r\n\x00not text'
        assert not new.exists()

    def test_rollback_goes_on_past_an_entry_that_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        restored = tmp_path / 'restored.py'
        restored.write_text('original', encoding='utf-8')
        stuck = tmp_path / 'stuck.py'
        removed = tmp_path / 'removed.py'
        journal = WriteJournal(tmp_path)
        journal.record_file(restored)
        restored.write_text('generated', encoding='utf-8')
        journal.record_file(stuck)
        journal.record_file(removed)
        removed.write_text('generated', encoding='utf-8')
        # By rollback time a non-empty directory stands where the run created a
        # file, and unlinking that fails (EISDIR, or EPERM off Linux). It sits
        # between the two entries that must still be undone.
        stuck.mkdir()
        (stuck / 'child.txt').write_text('x', encoding='utf-8')

        with caplog.at_level('WARNING', logger=JOURNAL_LOGGER):
            journal.rollback()

        assert restored.read_text(encoding='utf-8') == 'original'
        assert not removed.exists()
        assert stuck.is_dir()
        warned = [r.getMessage() for r in caplog.records if r.name == JOURNAL_LOGGER]
        assert len(warned) == 1
        assert str(stuck) in warned[0]

    def test_a_second_rollback_does_nothing(self, tmp_path: Path):
        target = tmp_path / 'models.py'
        target.write_text('original', encoding='utf-8')
        journal = WriteJournal(tmp_path)
        journal.record_file(target)
        target.write_text('generated', encoding='utf-8')
        journal.rollback()

        target.write_text('edited since', encoding='utf-8')
        journal.rollback()

        assert target.read_text(encoding='utf-8') == 'edited since'

    @POSIX_ONLY
    def test_rollback_restores_permission_bits(self, tmp_path: Path):
        target = tmp_path / 'models.py'
        target.write_text('original', encoding='utf-8')
        target.chmod(0o640)
        journal = WriteJournal(tmp_path)
        journal.record_file(target)

        # A write that replaces the file instead of truncating it loses the mode.
        target.unlink()
        target.write_text('generated', encoding='utf-8')
        target.chmod(0o644)
        journal.rollback()

        assert target.read_text(encoding='utf-8') == 'original'
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_a_file_that_cannot_be_read_raises_and_is_not_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = tmp_path / 'models.py'
        target.write_text('original', encoding='utf-8')
        real_read_bytes = Path.read_bytes

        def read_bytes(self: Path) -> bytes:
            if self == target:
                raise PermissionError(13, 'Permission denied', str(self))
            return real_read_bytes(self)

        monkeypatch.setattr(Path, 'read_bytes', read_bytes)
        journal = WriteJournal(tmp_path)

        with pytest.raises(PermissionError):
            journal.record_file(target)
        target.write_text('written anyway', encoding='utf-8')
        journal.rollback()

        assert target.read_text(encoding='utf-8') == 'written anyway'

    def test_a_directory_where_a_file_would_go_is_left_alone(self, tmp_path: Path):
        in_the_way = tmp_path / 'models.py'
        in_the_way.mkdir()
        (in_the_way / 'child.txt').write_text('x', encoding='utf-8')
        journal = WriteJournal(tmp_path)

        journal.record_file(in_the_way)
        journal.rollback()

        assert (in_the_way / 'child.txt').read_text(encoding='utf-8') == 'x'

    def test_ensure_dir_records_only_the_directories_it_creates(self, tmp_path: Path):
        package = tmp_path / 'pkg'
        package.mkdir()
        (package / 'keep.txt').write_text('keep', encoding='utf-8')
        journal = WriteJournal(tmp_path)

        journal.ensure_dir(package / 'a' / 'b')
        journal.ensure_dir(package)  # already there: nothing to record
        journal.rollback()

        assert [p.name for p in package.iterdir()] == ['keep.txt']
        assert (package / 'keep.txt').read_text(encoding='utf-8') == 'keep'

    def test_rollback_never_empties_a_directory(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """A file the journal did not record is somebody's: its directory stays."""
        journal = WriteJournal(tmp_path)
        journal.ensure_dir(tmp_path / 'a' / 'b')
        (tmp_path / 'a' / 'b' / 'stray.txt').write_text('x', encoding='utf-8')

        with caplog.at_level('WARNING', logger=JOURNAL_LOGGER):
            journal.rollback()

        assert (tmp_path / 'a' / 'b' / 'stray.txt').read_text(encoding='utf-8') == 'x'
        # ``a`` holds ``b``, so it cannot be removed either.
        assert (tmp_path / 'a' / 'b').is_dir()
        assert len([r for r in caplog.records if r.name == JOURNAL_LOGGER]) == 2

    def test_a_directory_somebody_else_made_meanwhile_is_not_ours_to_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = tmp_path / 'pkg'
        real_mkdir = Path.mkdir

        def lost_race(self: Path, *args, **kwargs) -> None:
            real_mkdir(self)  # the other process gets there first
            raise FileExistsError(17, 'File exists', str(self))

        monkeypatch.setattr(Path, 'mkdir', lost_race)
        journal = WriteJournal(tmp_path)

        journal.ensure_dir(target)
        journal.rollback()

        assert target.is_dir()

    def test_ensure_dir_refuses_a_file_like_mkdir_does(self, tmp_path: Path):
        (tmp_path / 'models.py').write_text('x', encoding='utf-8')

        with pytest.raises(FileExistsError):
            WriteJournal(tmp_path).ensure_dir(tmp_path / 'models.py')
        with pytest.raises(OSError):
            WriteJournal(tmp_path).ensure_dir(tmp_path / 'models.py' / 'sub')

    @POSIX_ONLY
    def test_rollback_never_follows_a_symlink(self, tmp_path: Path):
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'precious.txt').write_text('keep', encoding='utf-8')
        root = tmp_path / 'root'
        root.mkdir()
        created = root / 'created'
        journal = WriteJournal(root)
        journal.ensure_dir(created)
        (created / 'linked-dir').symlink_to(outside, target_is_directory=True)
        (created / 'linked-file').symlink_to(outside / 'precious.txt')

        journal.rollback()

        # The links are not the journal's, so their directory stays; what
        # matters is that nothing they lead to was touched.
        assert (created / 'linked-dir').is_symlink()
        assert (outside / 'precious.txt').read_text(encoding='utf-8') == 'keep'
        assert [p.name for p in outside.iterdir()] == ['precious.txt']

    @POSIX_ONLY
    def test_a_dangling_link_is_kept_and_the_file_made_through_it_removed(
        self, tmp_path: Path
    ):
        """Writing through the link creates its target; the link is the user's."""
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        link = tmp_path / 'models.py'
        link.symlink_to(elsewhere / 'models.py')
        journal = WriteJournal(tmp_path)

        journal.record_file(link)
        link.write_text('generated', encoding='utf-8')
        assert (elsewhere / 'models.py').read_text(encoding='utf-8') == 'generated'
        journal.rollback()

        assert link.is_symlink()
        assert not (elsewhere / 'models.py').exists()

    @POSIX_ONLY
    def test_a_link_to_a_file_is_restored_through_the_link(self, tmp_path: Path):
        real = tmp_path / 'real.py'
        real.write_text('original', encoding='utf-8')
        real.chmod(0o640)
        link = tmp_path / 'models.py'
        link.symlink_to(real)
        journal = WriteJournal(tmp_path)

        journal.record_file(link)
        link.write_text('generated', encoding='utf-8')
        journal.rollback()

        assert link.is_symlink()
        assert real.read_text(encoding='utf-8') == 'original'
        assert stat.S_IMODE(real.stat().st_mode) == 0o640


class TestContainment:
    """The journal deletes things, so it never acts outside the output directory.

    A bookkeeping mistake must not be able to turn rollback into deleting an
    ancestor of the output directory, a sibling, or the filesystem root. The
    journal deletes non-recursively, so the worst a mistake could cost is an
    empty directory; these tests pin that it cannot cost more.
    """

    def test_the_filesystem_root_is_not_a_valid_output_directory(self):
        with pytest.raises(OutputError, match='cannot be the root'):
            WriteJournal(Path(Path.cwd().anchor))

    @pytest.mark.parametrize('where', ['parent', 'sibling', 'dotdot'])
    def test_recording_a_path_outside_the_root_is_refused(
        self, tmp_path: Path, where: str
    ):
        root = tmp_path / 'root'
        root.mkdir()
        outside = {
            'parent': tmp_path,
            'sibling': tmp_path / 'other.py',
            'dotdot': root / '..' / 'other.py',
        }[where]
        journal = WriteJournal(root)

        with pytest.raises(OutputError, match='outside the output directory'):
            journal.record_file(outside)
        with pytest.raises(OutputError, match='outside the output directory'):
            journal.ensure_dir(outside)
        journal.rollback()

        assert tmp_path.is_dir() and root.is_dir()

    def test_a_corrupt_journal_deletes_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Whatever a bug records, undoing it must not leave the output directory."""
        root = tmp_path / 'root'
        root.mkdir()
        sibling = tmp_path / 'sibling'
        sibling.mkdir()
        (sibling / 'data.txt').write_text('keep', encoding='utf-8')
        precious = tmp_path / 'precious.txt'
        precious.write_text('keep', encoding='utf-8')
        anchor = Path(Path.cwd().anchor)
        journal = WriteJournal(root)
        journal._entries.extend(
            [
                _CreatedDir(anchor),
                _CreatedDir(tmp_path.parent),
                _CreatedDir(tmp_path),
                _CreatedDir(sibling),
                _CreatedFile(precious),
                _OverwrittenFile(precious, b'clobbered', None),
                _CreatedFile(root / '..' / 'precious.txt'),
            ]
        )

        with caplog.at_level('WARNING', logger=JOURNAL_LOGGER):
            journal.rollback()

        assert precious.read_text(encoding='utf-8') == 'keep'
        assert (sibling / 'data.txt').read_text(encoding='utf-8') == 'keep'
        assert tmp_path.is_dir() and root.is_dir() and anchor.is_dir()
        warned = [r for r in caplog.records if r.name == JOURNAL_LOGGER]
        assert len(warned) == 7
        assert all('outside the output directory' in r.getMessage() for r in warned)

    def test_the_root_is_removed_only_when_the_run_created_it(self, tmp_path: Path):
        existing = tmp_path / 'existing'
        existing.mkdir()
        journal = WriteJournal(existing)
        journal.ensure_dir(existing)
        journal.rollback()
        assert existing.is_dir()

        created = tmp_path / 'created'
        journal = WriteJournal(created)
        journal.ensure_dir(created)
        assert created.is_dir()
        journal.rollback()
        assert not created.exists()

    @POSIX_ONLY
    def test_a_link_out_of_the_root_is_not_written_through(self, tmp_path: Path):
        outside = tmp_path / 'outside.txt'
        outside.write_text('precious', encoding='utf-8')
        root = tmp_path / 'root'
        root.mkdir()
        (root / 'models.py').symlink_to(outside)
        journal = WriteJournal(root)

        with pytest.raises(OutputError, match='outside the output directory'):
            journal.record_file(root / 'models.py')

        assert outside.read_text(encoding='utf-8') == 'precious'

    @POSIX_ONLY
    def test_generation_refuses_to_overwrite_a_file_through_a_link_out_of_the_output(
        self, tmp_path: Path
    ):
        """A planted ``models.py -> ~/.bashrc`` must not be generated over."""
        outside = tmp_path / 'outside.txt'
        outside.write_text('precious', encoding='utf-8')
        output = tmp_path / 'client'
        output.mkdir()
        (output / 'models.py').symlink_to(outside)
        before = outside.stat()

        with pytest.raises(OutputError, match='outside the output directory'):
            _generate(_write_spec(tmp_path, VALID_SPEC), output)

        assert outside.read_text(encoding='utf-8') == 'precious'
        assert outside.stat().st_mtime_ns == before.st_mtime_ns
        assert (output / 'models.py').is_symlink()
        assert [p.name for p in output.iterdir()] == ['models.py']

    @POSIX_ONLY
    def test_a_path_swapped_for_a_link_out_of_the_root_is_not_followed_at_undo(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        outside = tmp_path / 'outside.txt'
        outside.write_text('precious', encoding='utf-8')
        root = tmp_path / 'root'
        root.mkdir()
        target = root / 'models.py'
        journal = WriteJournal(root)
        journal.record_file(target)  # the run is about to create it
        target.symlink_to(outside)  # ...and it is swapped before the rollback

        with caplog.at_level('WARNING', logger=JOURNAL_LOGGER):
            journal.rollback()

        assert outside.read_text(encoding='utf-8') == 'precious'
        assert target.is_symlink()
        assert any(
            'outside the output directory' in r.getMessage() for r in caplog.records
        )

    @POSIX_ONLY
    def test_a_link_inside_the_root_to_another_file_inside_it_is_fine(
        self, tmp_path: Path
    ):
        real = tmp_path / 'root' / 'real.py'
        real.parent.mkdir()
        real.write_text('original', encoding='utf-8')
        link = tmp_path / 'root' / 'models.py'
        link.symlink_to(real)
        journal = WriteJournal(tmp_path / 'root')

        journal.record_file(link)
        link.write_text('generated', encoding='utf-8')
        journal.rollback()

        assert real.read_text(encoding='utf-8') == 'original'
        assert link.is_symlink()


@pytest.fixture
def memory_root() -> Iterator[UPath]:
    """A scratch directory on fsspec's in-memory filesystem, a stand-in for a remote."""
    root = UPath(f'memory:///otterapi-journal-{uuid.uuid4().hex}')
    root.mkdir(parents=True)
    yield root
    root.rmdir(recursive=True)


class TestRemotePaths:
    def test_a_remote_path_is_journalled_without_permission_bits(
        self, memory_root: UPath
    ):
        existing = memory_root / 'old.py'
        existing.write_text('old', encoding='utf-8')
        journal = WriteJournal(memory_root)

        with activate(journal):
            ensure_dir(memory_root / 'pkg' / 'sub')
            write_text(memory_root / 'pkg' / 'sub' / 'new.py', 'new')
            write_text(existing, 'overwritten')
            touch(memory_root / 'pkg' / '__init__.py')
        assert (memory_root / 'pkg' / 'sub' / 'new.py').read_text() == 'new'
        journal.rollback()

        assert existing.read_text(encoding='utf-8') == 'old'
        assert not (memory_root / 'pkg').exists()

    def test_the_standalone_writers_work_on_a_remote_path(self, memory_root: UPath):
        generate_pagination_module(memory_root / 'client')

        assert (memory_root / 'client' / '_pagination.py').exists()


class TestHelpers:
    """The module-level helpers behind every write site."""

    def test_they_write_plainly_when_no_journal_is_active(self, tmp_path: Path):
        ensure_dir(tmp_path / 'a' / 'b')
        write_text(tmp_path / 'a' / 'b' / 'x.py', 'x = 1\n')
        touch(tmp_path / 'a' / '__init__.py')
        record_file(tmp_path / 'ignored.py')

        assert (tmp_path / 'a' / 'b' / 'x.py').read_text(encoding='utf-8') == 'x = 1\n'
        assert (tmp_path / 'a' / '__init__.py').exists()
        assert not (tmp_path / 'ignored.py').exists()

    def test_they_record_into_the_active_journal(self, tmp_path: Path):
        journal = WriteJournal(tmp_path)
        with activate(journal):
            ensure_dir(tmp_path / 'a' / 'b')
            write_text(tmp_path / 'a' / 'b' / 'x.py', 'x = 1\n')
            touch(tmp_path / 'a' / '__init__.py')
        journal.rollback()

        assert not (tmp_path / 'a').exists()

    def test_write_text_defaults_to_utf8_not_the_locale(self, tmp_path: Path):
        write_text(tmp_path / 'x.py', "name = 'café'\n")

        # Text mode writes the platform's line ending (CRLF on Windows), as the
        # ``open(path, 'w')`` this replaced did; the point here is the encoding.
        written = (tmp_path / 'x.py').read_bytes().replace(b'\r\n', b'\n')
        assert written == b"name = 'caf\xc3\xa9'\n"

    def test_activation_ends_with_the_block_even_on_error(self, tmp_path: Path):
        journal = WriteJournal(tmp_path)
        with pytest.raises(RuntimeError), activate(journal):
            raise RuntimeError('boom')

        write_text(tmp_path / 'after.py', 'x = 1\n')
        journal.rollback()

        assert (tmp_path / 'after.py').exists()

    def test_a_nested_activation_restores_the_outer_journal(self, tmp_path: Path):
        outer, inner = WriteJournal(tmp_path), WriteJournal(tmp_path)
        with activate(outer):
            with activate(inner):
                write_text(tmp_path / 'inner.py', 'x = 1\n')
            write_text(tmp_path / 'outer.py', 'x = 1\n')

        inner.rollback()
        assert not (tmp_path / 'inner.py').exists()
        assert (tmp_path / 'outer.py').exists()
        outer.rollback()
        assert not (tmp_path / 'outer.py').exists()

    def test_write_mod_is_journalled(self, tmp_path: Path):
        (tmp_path / 'pkg').mkdir()
        (tmp_path / 'pkg' / 'mod.py').write_text('x = 0\n', encoding='utf-8')
        journal = WriteJournal(tmp_path)
        with activate(journal):
            write_mod(
                ast.parse('x = 1').body, tmp_path / 'pkg' / 'mod.py', format_code=False
            )
            write_mod(
                ast.parse('y = 2').body, tmp_path / 'pkg' / 'new.py', format_code=False
            )
        assert (tmp_path / 'pkg' / 'new.py').exists()
        journal.rollback()

        assert [p.name for p in (tmp_path / 'pkg').iterdir()] == ['mod.py']
        assert (tmp_path / 'pkg' / 'mod.py').read_text(encoding='utf-8') == 'x = 0\n'

    def test_the_standalone_module_writers_are_journalled(self, tmp_path: Path):
        # The directories predate the run, so a file written behind the
        # journal's back could not be swept up with a directory it created.
        for name in ('one', 'two', 'three'):
            (tmp_path / name).mkdir()
        journal = WriteJournal(tmp_path)
        with activate(journal):
            generate_pagination_module(tmp_path / 'one')
            generate_export_module(tmp_path / 'two')
            PaginationFeature().write(tmp_path / 'three')
        assert (tmp_path / 'one' / '_pagination.py').exists()
        assert (tmp_path / 'two' / '_export.py').exists()
        assert (tmp_path / 'three' / '_pagination.py').exists()
        journal.rollback()

        assert [list(p.iterdir()) for p in sorted(tmp_path.iterdir())] == [[], [], []]
