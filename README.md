# Sinthesizing facial expressions using Generative Adversarial Networks
## Author: Rodrigo Parra

This repository contains the source code for the model and experiments
corresponding to my master thesis developed under supervision of Youssef Kashef
and Prof. Dr. Klaus Obermayer from the Neural Information Processing Group (NI) at
TU Berlin.

## Important files

- **main.py**: executable file that allows to run a single instance of the experiment, i.e.
  training a conditional GAN and evaluating it given a certain level of imbalance, label, etc.
  
- **model.py**: code corresponding to the AC-GAN.

- **keras_inception_classifier.py**: code corresponding to the fine-tuned model used
to evaluate GAN sampling as an imbalance remedy.

- **run_experiments.sh**: bash script that allows bulk run of experiments given certain parameters.

- ***\*.ipynb***: Jupyter notebooks used for analysis of the results. 


## How to run

To run a batch of experiments consecutively with the same parameters execute the following command:

``
nohup bash ../run_experiments.sh celeba [number of runs] [number of runs already finished] [imbalance level] [cache folder] [dataset folder] [label] [gpu capacity to use] &
``

An example of the previous command with actual parameters looks like: 
 
``
nohup bash ../run_experiments.sh celeba 3 0 0.175 /mnt/antares_raid/home/rparra/workspace/DCGAN-tensorflow/cache_local /mnt/antares_raid/home/rparra/workspace/DCGAN-tensorflow/data_local/celebA Mouth_Slightly_Open 0.5 &
``

## Results folder structure

Jupyter notebooks used for analysis require a certain folder structure to be ran as is.

### Results CelebA.ipynb
    results/
        attractive/
            results_celeba_Attractive_0.1/
            results_celeba_Attractive_0.25/
            .
            .
            .
        high_cheekbones/
        lipstick/
        mouth_slightly_open/
        smiling/
        
Where *results_celeba_Attractive_0.1* and analogous folders store 
the results of running *run_experiments.sh* for the *Attractive* label
and *0.1* level of imbalance.

### UMAP.ipynb

    img_align_celeba/
    cache/
    cache_generated/
    
Where **img_align_celeba** contains the CelebA dataset, *cache* contains the cached
splits corresponding to a single run of the training procedure and
*cache_generated* contains images sampled from the trained GAN.


##Credit

This project was based on the DCGAN implementation available in [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow),
from which it was forked.