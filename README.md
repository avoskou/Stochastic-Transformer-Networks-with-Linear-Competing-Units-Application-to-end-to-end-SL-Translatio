# Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end Sign Language Translation 


Automating sign language translation (SLT) is a challenging
real-world application. Despite its societal importance,
though, research progress in the field remains rather
poor. Crucially, existing methods that yield viable performance
necessitate the availability of laborious to obtain
gloss sequence groundtruth. In this paper, we attenuate
this need, by introducing an end-to-end SLT model that does
not entail explicit use of glosses; the model only needs text
groundtruth. This is in stark contrast to existing end-to-
end models that use gloss sequence groundtruth, either in
the form of a modality that is recognized at an intermedi-
ate model stage, or in the form of a parallel output process,
jointly trained with the SLT model. Our approach constitutes
a Transformer network with a novel type of layers that
combines: (i) local winner-takes-all (LWTA) layers with
stochastic winner sampling, instead of conventional ReLU
layers, (ii) stochastic weights with posterior distributions
estimated via variational inference, and (iii) a weight com-
pression technique at inference time that exploits estimated
posterior variance to perform massive, almost lossless com-
pression. We demonstrate that our approach can reach the
currently best reported BLEU-4 score on the PHOENIX
2014T benchmark, but without making use of glosses for
model training, and with a memory footprint reduced by
more than 70%

The code is based on:
1. Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation.
2. Joey NMT (https://github.com/joeynmt/joeynmt) 
3. Nonparametric Bayesian Deep Networks with Local Competition
4. Bayesian Compression for Deep Learning


## Reference

Please cite :

@inproceedings{voskou2021stochastic,
  title={Stochastic transformer networks with linear competing units: Application to end-to-end sl translation},
  author={Voskou, Andreas and Panousis, Konstantinos P and Kosmopoulos, Dimitrios and Metaxas, Dimitris N and Chatzis, Sotirios},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11946--11955},
  year={2021}
}


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



## Acknowledgement
This  research  was  partially  supported  by  the  ResearchPromotion  Foundation  of  Cyprus,  through  the  grant:  INTERNATIONAL/USA/0118/0037  (Dimitrios  Kosmopoulos, Dimitris Metaxas), and the European Union’s Horizon2020 research and innovation program, under grant agreement  No  872139,  project  aiD  (Andreas  Voskou,  Sotirios Chatzis)
