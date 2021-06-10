#  Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end SL Translation


The code is based on:
1. Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation.
2. Joey NMT (https://github.com/joeynmt/joeynmt) 
3. Nonparametric Bayesian Deep Networks with Local Competition
4. Bayesian Compression for Deep Learning

## Requirements
* python 3.6+
* Download the feature files using the `data/download.sh` script.
* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

Tested on a single GPU (not tested on CPU or multiple GPUs ).



## Usage

To train a model:

  `python -m signjoey train configs/example.yaml`
  
To test  an excisting model:
  
  `python -m signjoey test configs/example.yaml`
  


Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.


