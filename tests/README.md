# Testing

```bash
python -m pytest -q                           # one line per file
python -m pytest -v                           # one line per test
python -m pytest tests/test_reduce.py         # one file
python -m pytest -k sentinel                  # name filter
python -m pytest -k "not round_trip" -x       # skip some, stop on first failure
python -m pytest --lf                         # rerun only last failures
```
