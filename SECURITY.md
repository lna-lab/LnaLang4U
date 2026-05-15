# Security

## Reporting security issues

If you discover a security vulnerability, please do NOT open a public GitHub issue.

Instead, contact the maintainers privately via email at `lna-lab-security@googlegroups.com` or reach out through the [Lna-Lab website](https://lna-lab.com).

## Best practices

- Do not post private model tokens, API keys, or local secrets in issues or PRs.
- The `diskkv/` directory may contain cached KV data from processed prompts. Treat it as potentially sensitive.
- Docker containers run with GPU access. Review the Dockerfile and launch commands before executing on sensitive infrastructure.
