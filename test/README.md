Testing

1. Install requirements:
```
pip install -r test/requirements.txt
```

2. Run tests

Some tests are too slow. You can run fast tests and then slow tests.

To run only fast tests:
```
pytest test/ -p no:warnings -m "not slow"
```
To run only slow tests:
```
pytest test/ -p no:warnings -m "slow"
```
