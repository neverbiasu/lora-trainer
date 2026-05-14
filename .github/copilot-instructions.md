Always enforce the repository quality gate before any push.

Required pre-push checks:
- `ruff check .`
- `black --check .`
- `pytest tests -q`

If any check fails, fix the failures and rerun the full gate until all checks pass.
Do not push when any of the checks are red.

When a change is intended for push, run the full gate proactively without waiting
for the user to ask.