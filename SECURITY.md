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
