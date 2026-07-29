# Vendored real-world OpenAPI specs (gzipped)

Large, messy, production OpenAPI documents used by the slow smoke test
(`test_real_specs_smoke.py`) to prove OtterAPI can generate a valid, importable
client end to end -- the case the small curated fixtures cannot cover.

Stored gzipped to keep the repository small (~1.6 MB vs ~21 MB uncompressed);
the test decompresses them in-memory.

| File | Source |
|------|--------|
| `stripe.openapi.json.gz` | https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json |
| `github.openapi.json.gz` | https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json |
