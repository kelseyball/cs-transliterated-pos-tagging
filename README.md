This is the code and data for the experiments in ["Part-of-Speech Tagging for Code-Switched, Transliterated Texts without Explicit Language Identification"](http://aclweb.org/anthology/D18-1347). (EMNLP 2018)

## Install Dependencies
  - [DyNet version 2.0](https://github.com/clab/dynet)
  - [indictrans transliteration module](https://github.com/libindic/indic-trans)
  
## Download pre-trained word embeddings
Clone the [indic-word2vec-embeddings] repository (https://bitbucket.org/kelseyball/indic-word2vec-embeddings/src/master/) into the data/raw directory.
```git clone https://kelseyball@bitbucket.org/kelseyball/indic-word2vec-embeddings.git data/raw/
```
  
## Data prep
`python prep.py`
This script pre-transliterates the Hindi training data and word embeddings into Latin. The generated files are placed in the data/clean directory and used for the baseline experiment.

## Experiments
The baseline and experimental models are in the baseline.py and our-model.py files, respectively. The experiments listed below and results are described in greater detail in our paper.

- Baseline
```python baseline.py --htrain data/clean/hi_roman-ud-train.conllu  --etrain data/raw/en-ud-train.conllu --cdev data/clean/TWEETS-dev-v2-unsup-dist.conll --hi-embds data/clean/Hindi_roman.vec --en-embds data/raw/indic-word2vec-embeddings/English.vec --hi-limit 50000 --en-limit 50000 --iter 20
```
- Our model (multi-lingual)
```python our-model.py --htrain data/raw/hi-ud-train.conllu  --etrain data/raw/en-ud-train.conllu --cdev data/clean/TWEETS-dev-v2-unsup-dist.conll --hi-embds data/raw/indic-word2vec-embeddings/Hindi_utf.vec --en-embds data/raw/indic-word2vec-embeddings/English.vec --hi-limit 50000 --en-limit 50000 --iter 20
```
- Our model (forced language choice)
```python our-model.py --htrain data/raw/hi-ud-train.conllu  --etrain data/raw/en-ud-train.conllu --cdev data/clean/TWEETS-dev-v2-unsup-dist.conll --hi-embds data/raw/indic-word2vec-embeddings/Hindi_utf.vec  --en-embds data/raw/indic-word2vec-embeddings/English.vec --hi-limit 50000 --en-limit 50000 --iter 20 --use-ltags
```
- Our model (languages weighted by HMM)
```python our-model.py --htrain data/raw/hi-ud-train.conllu --etrain data/raw/en-ud-train.conllu --cdev data/clean/TWEETS-dev-v2-unsup-dist.conll --hi-embds data/raw/indic-word2vec-embeddings/Hindi_utf.vec  --en-embds data/raw/indic-word2vec-embeddings/English.vec --hi-limit 50000 --en-limit 50000 --iter 20
```
- Our model (oracle language choice)
```python our-model.py --htrain data/raw/hi-ud-train.conllu  --etrain data/raw/en-ud-train.conllu --cdev data/raw/TWEETS-dev-v2.conll --hi-embds data/raw/indic-word2vec-embeddings/Hindi_utf.vec  --en-embds data/raw/indic-word2vec-embeddings/English.vec --hi-limit 50000 --en-limit 50000 --iter 20 --use-ltags
```
