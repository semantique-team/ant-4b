# Contributing to ANT-4B

Thanks for your interest in contributing. ANT-4B is open source, but it is actively
maintained by the Semantique Team and every change goes through us before it lands.

## Ground rules

- `main` is protected: **nobody pushes or commits directly to it** — not even the
  maintainers. Every change lands through a pull request.
- A pull request is merged **only after approval by at least one Semantique maintainer**
  (a code owner listed in `.github/CODEOWNERS`). Reviews from other contributors are
  welcome and useful, but they do not approve a merge.
- Keep pull requests small and focused. Unrelated refactors go in their own PR.
- We may close issues or PRs that are out of scope, unmotivated, or that change the
  public API without prior discussion.

## How to contribute

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b my-change
   ```
2. Make your change, with tests if it changes behavior.
3. Open a pull request against `main`. Describe **what** the change does and **why**,
   and how to verify it.
4. Wait for review. Maintainers may request changes; once a maintainer approves, a
   maintainer will merge. Do not merge your own PR.

For large changes (new API surface, checkpoint format changes, training pipeline
changes), open an issue first so we can agree on the design.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Integration tests that download a real backbone are opt-in:

```bash
ANT_TEST_QWEN=Qwen/Qwen3.5-4B pytest -m integration
```

## Reporting bugs and requesting features

Use [GitHub Issues](../../issues). For bugs, include the checkpoint, the command you
ran, and the full traceback. Security-sensitive reports: contact the maintainers
privately instead of opening a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[Apache-2.0 license](LICENSE) of this repository.
