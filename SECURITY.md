# Security Policy

Thanks for helping keep **otterapi** and its users safe.

otterapi is a code generator: it reads OpenAPI documents and emits Python HTTP
clients. Security issues can therefore live in two places — in the generator
itself, and in the code it produces. Both are in scope (see below).

## Supported versions

otterapi is pre-1.0 and ships from `main`. Security fixes are applied to the
latest released version on PyPI only. Please make sure you can reproduce an
issue on the most recent release before reporting.

| Version        | Supported          |
| -------------- | ------------------ |
| latest release | :white_check_mark: |
| older releases | :x:                |

## Reporting a vulnerability

**Please do not open a public issue, discussion, or pull request for security
vulnerabilities.** Public disclosure before a fix is available puts users at
risk.

Instead, report privately using GitHub's **private vulnerability reporting**:

1. Go to the [Security advisories page](https://github.com/danplischke/otterapi/security/advisories/new).
2. Click **Report a vulnerability**.
3. Fill in the details (see below).

If you are unable to use GitHub's private reporting, you may contact the
maintainer directly at the email associated with the
[@danplischke](https://github.com/danplischke) account.

### What to include

A good report helps us triage quickly:

- A description of the vulnerability and its potential impact.
- The otterapi version, Python version, and OS.
- A minimal OpenAPI document and/or the `otter` command that triggers it.
- The generated output or traceback, if relevant.
- Any proof-of-concept, reproduction steps, or suggested fix.

## What to expect

- **Acknowledgement** within 3 business days.
- An initial **assessment** (severity + whether we can reproduce) within 7
  business days.
- We'll keep you updated on remediation progress and coordinate a disclosure
  timeline with you. We aim to ship a fix before any public disclosure.
- With your permission, we're happy to credit you in the release notes and
  advisory once the issue is resolved.

## Scope

**In scope**

- Vulnerabilities in the otterapi generator (e.g. code injection via a crafted
  OpenAPI document, path traversal when writing generated files, unsafe
  handling of `$ref` or external references).
- Generated code that is insecure by default (e.g. disabled TLS verification,
  leaking credentials, unsafe deserialization) for a well-formed spec.

**Out of scope**

- Vulnerabilities in third-party dependencies — please report those upstream
  (though a heads-up is welcome; we'll bump the pin).
- Insecurity that originates entirely from a user's own OpenAPI document or
  hand-written `client.py` customizations.
- Findings that require a malicious local environment or already-compromised
  machine.

## Handling untrusted specs

otterapi does **not** execute the OpenAPI documents it reads, but generation
does construct Python source from spec-controlled values. Treat OpenAPI
documents from untrusted sources as untrusted input, review generated code
before running it, and prefer generating in an isolated environment.

What the generator does about them:

- **Source:** names that become Python identifiers are reduced to
  `[A-Za-z0-9_]`; every other spec string is emitted as a string literal. A path
  that does not start with `/`, contains a `.`/`..` segment, or has a
  `{placeholder}` with no matching parameter fails generation.
- **Files:** module names derived from tags and paths are reduced to
  `[a-z0-9_]` and renamed if they would collide with a file the generator owns
  (`client`, `models`, `__init__`, the runtime helpers). The generator refuses to
  write through a symlink that leaves the output directory, and a failed run
  undoes only what it wrote.
- **Loading:** downloads are capped at 50 MiB after decompression and 60 seconds,
  redirects onto private or loopback addresses are refused, and YAML alias bombs
  are rejected. The address check resolves names before connecting, so DNS
  rebinding is not covered; fetch specs from hosts you trust, or save them to a
  file first.
- **Defaults:** a relative server URL is resolved without the credentials or API
  key in the URL the spec came from, server URLs that are not `http(s)` are
  rejected, and a Swagger 2.0 document without `schemes` defaults to `https`.

## Untrusted data in generated clients

A generated client treats what the API returns, and the values callers pass as
path parameters, as untrusted:

- Every model's `_repr_html_` (used by Jupyter) escapes its values, so a string in
  a response cannot inject markup into a notebook.
- A path parameter that is `None`, empty, `.` or `..` raises instead of being
  sent, and a request path that does not start with `/` or contains a dot segment
  is refused, so a value or a spec path cannot retarget a request or change its
  host.
- Credentials are read from environment variables prefixed with the output
  package's name (not a prefix shared by every client), and with an empty prefix
  only the schemes named in `auth.env_vars` read the environment.
- `POST` and `PATCH` are not retried after a timeout or a 5xx, which could repeat
  a request the server already acted on; `Retry-After` is capped.
- CSV and TSV exports neutralise cells that a spreadsheet would run as formulas.
- Cursor pagination stops with an error when the server returns a cursor it
  already returned.

A case where one of these does not hold is in scope as "generated code that is
insecure by default".
