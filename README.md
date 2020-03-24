# Bengali.AI Bronze Medal Implementation

# Overview

This code is for Bengali.AI that is Kaggle competition for recognize handwritten Bengali. (https://www.kaggle.com/c/bengaliai-cv19)
We joined this competition and won Bronze Medal(164th / 2059). We first implemented and did experiment with Colaboratory notebook but after the competition I transplanted that notebook to scripts.

![Bengali.AI](https://user-images.githubusercontent.com/18682053/77383028-3f86b880-6dc5-11ea-99e9-9aaff0670a84.png "https://www.kaggle.com/c/bengaliai-cv19")


## Details 
We use pretrained *Efficient-net b3* for feature extraction and finetuned that. Our model mainly predict not 'grapheme' directly but 'grapheme root', 'vowel diacritic' and 'consonant diacritic' which consist of 'grapheme'. TensorBoard logger is implemented for monitoring experiments and you can watch your results easily.

![TensorBoard](https://user-images.githubusercontent.com/18682053/77383034-41507c00-6dc5-11ea-816a-bfc8cb5fbcc5.png)

## Requirement
I'm sorry. Will write later...

## Usage
Please run \*\*\*\*.sh with your parameters.  
You can use Single or Multi GPU.

## Future 
I have not transplanted my notebook completely about augment hyperparameters. I'm going to do this soon!
