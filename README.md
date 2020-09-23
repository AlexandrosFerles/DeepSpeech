# Automatic Speech Recognition for the Swedish Language

This repository contains work for the project 'Automatic Speech Recognition for the Swedish Language' as part of the course 'DT2112 Speech Technology'.  
  
Based on a fork from Mozilla's [implementation](https://github.com/mozilla/DeepSpeech) of the [Deep Speech](https://arxiv.org/abs/1412.5567) architecture, we provide an implementation of this network for the Swedish language, applied on data drawn from the NST dataset. All experiments were ran in an Ubuntu machine with an NVIDIA 2080 Ti graphics card.

## Datasets

For our work, we used two subsets of the NST dataset. The first dataset is quite small in size while the second one is bigger in size, pushing the GPU card to its limits. Experiments in the full dataset could not be conducted due to GPU memory errors. For both datasets you need separate csv files for the train, validation and test set. If you wish to use the exact same data as we did, you can use the csv files added on the 'csv_files' folder but you need to update the path destination for each file listed in the csv cells. 

## N-gram model

For training and inference, an N-gram model, along with a text file for the characters of the Swedish alphabet. This is provided in this repo and can be used directly, but you can create your own and use them by modifying the file names and path locations. 

## How to train

In general you can define several training hyper-paramaters like batch size, amount of learning rate applied and dropout. Some sample executable training files can be found in the 'script_files' folder which you can move on the repo's top level and use directly, or use them as a basis and change the configurations. For your convenience, we provide the files 'small_dataset.sh', 'small_dataset_dropout.sh' (which shows how different amount of the dropout rate can be applied in each fully connected layer) and 'big_dataset.sh' respectively, to recreate some of the best results that we got. Checkpoints are stored in the 'results' folder, more specifically in the 'checkout' and 'model_export' subfolders. To initiate a training process from scratch, you need to empty these folders, either saving their contents to a different location in order to use the checkpointed model again if you wish or by simply deleting them.

## How to use a pre-trained model

Place the checkpointed files in the 'results/checkout' and 'results/moder_export' paths and simpy initiate a training process with your new configurations. If you use the same configuration as those that created the checkpointed model, you will essentially either reproduce the inference process on the test set or train the model for more training epochs if early stopping was not activated.  

You can find more information on our project [report](https://github.com/AlexandrosFerles/DeepSpeech/blob/master/report/Automatic_Speech_Recognition_of_the_Swedish_Language_Project_Report.pdf).

# Μembers 

This work was completed with the contribution of [Anastasios Lamproudis](https://github.com/TLampr) and [Leonidas Valavanis](https://github.com/valavanisleonidas) whom I deeply thank. 
