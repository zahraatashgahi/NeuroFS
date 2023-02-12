# NeuroFS

This repository contains code for the paper, "Supervised Feature Selection with Neuron Evolution in Sparse Neural Networks" by Zahra Atashgahi, Xuhao Zhang, Neil Kichler, Shiwei Liu, Lu Yin, Mykola Pechenizkiy, Raymond Veldhuis, Decebal Constantin Mocanu (https://openreview.net/forum?id=GcO6ugrLKp).

### Methodology
![algorithm](https://github.com/zahraatashgahi/NeuroFS/blob/main/NeuroFS.JPG?raw=true)


### Feature Selection on the MNIST dataset
![MNIST](https://github.com/zahraatashgahi/NeuroFS/blob/main/mnist.JPG?raw=true)



### How to run
```
K=50
seed=0 
dataset=coil20 # USPS isolet  har MNIST Fashion-MNIST BASEHOCK (batch size = 100) arcene Prostate_GE SMK GLA-BRA-180 (batch size = 20) 

python3 ./code/main.py   --dataset_name $dataset \
		--model "NeuroFS" --K $K\
		--batch_size 100 --lr 0.01 --epochs 100\
		--zeta_in 0.2 --zeta_hid 0.3 --epsilon 30\
		--num_hidden 1000 --seed $seed --wd 0.00001\
		--gradient_addition --frac_epoch_remove 0.65 \
		--activation "tanh" 
```



### Requirements
 Following Python packages have to be installed before executing the project code:
```
keras                     2.3.1           
keras-gpu                 2.3.1                   
matplotlib                3.5.1              
numpy                     1.21.5         
python                    3.7.13            
python-dateutil           2.8.2              
scikit-learn              1.0.2                 
scipy                     1.7.3          
tensorflow                1.14.0            
tensorflow-gpu            1.14.0                  
```


### Acknowledgements
Starting of the code is the Sparse Evolutionary Training (SET) algorithm which is available at: https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks



### Citation
```
@article{
atashgahi2022supervised,
title={Supervised Feature Selection with Neuron Evolution in Sparse Neural Networks},
author={Zahra Atashgahi and Xuhao Zhang and Neil Kichler and Shiwei Liu and Lu Yin and Mykola Pechenizkiy and Raymond Veldhuis and Decebal Constantin Mocanu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=GcO6ugrLKp},
note={}
}

```

### Contact
email: z.atashgahi@utwente.nl
