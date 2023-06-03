# MDLRNN-torch 

## PyTorch port of Minimum Description Length Recurrent Neural Networks (MDLRNNs)

MDLRNNs are recurrent neural nets that optimize the MDL objective, an approximation of Kolmogorov complexity.
To do this we minimize the encoding length of the network's architecture itself, alongside the classical loss.

Optimizing the encoding length makes it possible to prevent overfitting and reach perfect generalizations for many formal languages that aren't learned well by classical models, such as `aⁿbⁿ` and `Dyck-1`.

It also results in very small networks, such as this `aⁿbⁿ` RNN that only has one hidden unit that serves as a counter:

** #TODO image **

## Citing this work

Please use this BibTeX if you wish to cite this project in your publication:

```
@article{Lan-Geyer-Chemla-Katzir-MDLRNN-2022,
  title = {Minimum Description Length Recurrent Neural Networks},
  author = {Lan, Nur and Geyer, Michal and Chemla, Emmanuel and Katzir, Roni},
  year = {2022},
  month = jul,
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {10},
  pages = {785--799},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00489},
}
```

## Pre-trained models

`models` contains pre-trained networks that are 100% accurate for the following formal languages:
- aⁿbⁿ
- aⁿbⁿcⁿ
- aⁿbᵐcⁿ⁺ᵐ
- Dyck-1

**#TODO table with num hidden units**


More coming soon!

## Fine-tuning using backprop

The networks use only differentiable activations (relu, tanh) and a final softmax layer, so they can be fine-tuned using regular backpropagation.  

```
# TODO example
- load checkpoint
- define loss, train
```

## Evolving networks from scratch for full MDL

MDLRNNs optimize the Minimum Description Length objective, which isn't differentiable. We use a genetic algorithm to train them and then port them to PyTorch. 

To evolve networks from scratch using a genetic algorithm, use the [genetic algorithm MDLRNN trainer](https://github.com/taucompling/mdlrnn) trainer here:
https://github.com/taucompling/mdlrnn/
