# Automatic Speech Recognition for the Swedish Language

This repository contains work for the project 'Automatic Speech Recognition for the Swedish Language' as part of the course 'DT2112 Speech Technology'.  
  
Based on a fork from Mozilla's [implementation](https://github.com/mozilla/DeepSpeech) of the [Deep Speech](https://arxiv.org/abs/1412.5567) architecture, we provide an implementation of this network for the Swedish language, applied on data drawn from the NST dataset. All experiments were ran in an Ubuntu machine with an NVIDIA 2080 Ti graphics card.

## Datasets

For our work, we used two subsets of the NST dataset. The first dataset is quite small in size while the second one is bigger in size, pushing the GPU card to its limits. Experiments in the full dataset could not be conducted due to GPU memory errors. For both datasets you need separate csv files for the train, validation and test set. If you wish to use the exact same data as we did, you can use the csv files added on the 'csv_files' folder but you need to update the path destination for each file listed in the csv cells. 

## How to train

In general you can define several training paramaters like batch size, amount of learning rate applied and dropout. Some sample executable training files can be found in the 'script_files' folder which you can move on the repo's top level and use directly, or use them as a basis and change the configurations. 
