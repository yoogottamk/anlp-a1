# ANLP A1

## Initial setup
### setup
`virtualenv` HIGHLY recommended.
```
virtualenv -p python3.8 venv
source venv/bin/activate

pip install -r requirements.txt
```

### Downloading files
From the repo root, run 
```
./prepare-inference.sh
```
That should download 4 files.

### Setting PYTHONPATH
Then, run
```
source .env
```
This should set your PYTHONPATH

## How To
### Run COM Vectorizer
Run
```
python -m anlp_a1.com
```

You should get a python console with some instructions.

### Run CBOW Vectorizer
Run
```
python -m anlp_a1.cbow
```

You should get a python console with some instructions.
