# Bengali.AI Bronze Medal Implementation

# Overview

This code is for Bengali.AI that is Kaggle competition for recognize handwritten Bengali. (https://www.kaggle.com/c/bengaliai-cv19)
We joined this competition and won Bronze Medal(164th / 2059). We first implemented and did experiment with Colaboratory notebook but after the competition I transplanted that notebook to scripts.

## Details 
We use pretrained *Efficient-net b3* for feature extraction and finetuned that. Our model mainly predict not 'grapheme' directly but 'grapheme root', 'vowel diacritic' and 'consonant diacritic' which consist of 'grapheme'. Tensorboard logger is implemented for monitoring experiments and you can watch your results easily.

## Requirement
I'm sorry. Will write later...

## Usage
Please run \*\*\*\*.sh with your parameters.  
You can use Single or Multi GPU.

## Future 
I have not transplanted my notebook completely about augment hyperparameters. I'm going to do this soon!
