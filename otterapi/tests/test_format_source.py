"""``format_source`` must give the same result whatever the platform's locale.

It pipes the module through ``ruff format -``.  In text mode Python encodes the
input with the locale's encoding, which on Windows is cp1252: the em dash in the
generated client's docstrings became a byte that ruff rejects as invalid UTF-8,
``format_source`` quietly returned the source unformatted, and a long import
line came out unwrapped -- an I001 finding in the user's own linter.  The
Windows CI jobs caught it only because a new import happened to be long enough.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from otterapi.codegen.utils import format_source

# What Python's text mode would use for the pipe on a Windows machine.
WINDOWS_LOCALE = 'cp1252'


def _fake_ruff(returncode: int = 0, crlf: bool = False):
    """A stand-in for ``subprocess.run`` that behaves like ruff does on stdin.

    ruff reads UTF-8 and nothing else.  Given text, the stand-in encodes it with
    the Windows locale first, as ``subprocess`` would.
    """
    calls: list[dict] = []

    def run(cmd, input=None, capture_output=False, text=False, encoding=None, **kw):
        calls.append({'cmd': cmd, 'input': input, 'text': text, 'encoding': encoding})
        data = (
            input.encode(encoding or WINDOWS_LOCALE)
            if isinstance(input, str)
            else input
        )
        try:
            decoded = data.decode('utf-8')
        except UnicodeDecodeError:
            return subprocess.CompletedProcess(cmd, 2, b'', b'error: invalid UTF-8')
        out = ('# formatted\n' + decoded).encode('utf-8')
        if crlf:
            out = out.replace(b'\n', b'\r\n')
        if returncode:
            return subprocess.CompletedProcess(cmd, returncode, b'', b'error')
        if text or encoding:
            return subprocess.CompletedProcess(cmd, 0, out.decode('utf-8'), '')
        return subprocess.CompletedProcess(cmd, 0, out, b'')

    run.calls = calls  # type: ignore[attr-defined]
    return run


class TestEncoding:
    @pytest.mark.parametrize(
        'text',
        [
            'plain ascii',
            # In cp1252, so the old text-mode call sent a byte ruff rejects.
            'await something — for example refreshing a token',
            'café …',
            # Not in cp1252 at all: the old call raised instead of falling back.
            'arrow → here',
            'otter \U0001f9a6',
        ],
    )
    def test_non_ascii_source_reaches_ruff_as_utf8(self, monkeypatch, text):
        monkeypatch.setattr(subprocess, 'run', _fake_ruff())

        result = format_source(f'x = 1\n"""{text}"""\n')

        assert result.startswith('# formatted\n'), 'ruff rejected the input'
        assert text in result

    def test_the_input_is_bytes_not_locale_dependent_text(self, monkeypatch):
        fake = _fake_ruff()
        monkeypatch.setattr(subprocess, 'run', fake)

        format_source('x = 1\n')

        (call,) = fake.calls
        assert call['cmd'] == ['ruff', 'format', '-']
        assert isinstance(call['input'], bytes)
        assert not call['text'] and call['encoding'] is None


class TestOutput:
    def test_line_endings_from_ruff_are_normalised(self, monkeypatch):
        """The file is written in text mode, which adds its own on Windows."""
        monkeypatch.setattr(subprocess, 'run', _fake_ruff(crlf=True))

        result = format_source('x = 1\ny = 2\n')

        assert '\r' not in result
        assert result.startswith('# formatted\nx = 1\ny = 2')

    def test_blank_line_pass_still_runs(self, monkeypatch):
        monkeypatch.setattr(subprocess, 'run', _fake_ruff())
        source = 'if TYPE_CHECKING:\n    import x\n__all__ = ()\n'

        assert '    import x\n\n__all__' in format_source(source)


class TestFallbacks:
    def test_a_failing_ruff_leaves_the_source_alone(self, monkeypatch):
        monkeypatch.setattr(subprocess, 'run', _fake_ruff(returncode=2))

        assert format_source('x = 1\n') == 'x = 1\n'

    def test_a_missing_ruff_leaves_the_source_alone(self, monkeypatch):
        def missing(*args, **kwargs):
            raise FileNotFoundError('ruff')

        monkeypatch.setattr(subprocess, 'run', missing)

        assert format_source('x = 1\n') == 'x = 1\n'

    def test_source_that_cannot_be_encoded_is_returned_unformatted(self, monkeypatch):
        monkeypatch.setattr(subprocess, 'run', _fake_ruff())

        # A lone surrogate has no UTF-8 encoding.
        assert format_source('x = "\ud800"\n') == 'x = "\ud800"\n'


@pytest.mark.skipif(shutil.which('ruff') is None, reason='ruff is not installed')
def test_real_ruff_wraps_a_long_import_next_to_non_ascii_text():
    """End to end: the case that failed on Windows, with the real formatter."""
    source = (
        'from ._retry import _backoff_sleep, _backoff_sleep_async, '
        '_should_retry_error, _should_retry_response\n'
        '\n\n'
        'def f():\n'
        '    """Wait — then retry."""\n'
    )

    result = format_source(source)

    assert 'from ._retry import (\n    _backoff_sleep,' in result
    assert 'Wait — then retry.' in result
