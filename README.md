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

## Misc
### generating gensim embeddings
This was done on colab because of faster internet
```python
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
wv.most_similar("camera", topn=10)
```

```
[('cameras', 0.8131939172744751),
 ('Wagging_finger', 0.7311819791793823),
 ('camera_lens', 0.7250816822052002),
 ('camcorder', 0.7037474513053894),
 ('Camera', 0.6848659515380859),
 ('Canon_digital_SLR', 0.6474252939224243),
 ('Cameras', 0.6350969076156616),
 ('Nikon_D####_digital_SLR', 0.6259366273880005),
 ('tripod', 0.6189837455749512),
 ('EyeToy_USB', 0.6173486709594727)]
```
