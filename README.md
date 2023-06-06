# MDLRNN-torch 

## PyTorch port of Minimum Description Length Recurrent Neural Networks (MDLRNNs)

MDLRNNs are recurrent neural nets that optimize the MDL objective, an approximation of Kolmogorov complexity.
To do this we minimize the encoding length of the network's architecture itself, alongside the classical loss.

Optimizing for encoding length makes it possible to prevent overfitting and reach perfect generalizations for many formal languages that aren't learned well by classical models, such as `aⁿbⁿ` and `Dyck-1`.

It also results in very small networks, such as this `aⁿbⁿ` RNN cell that only has one hidden unit that serves as a counter:

<img src="assets/an_bn.png" width="430px" style="margin: 15px 0 5px 0"> 

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

`./models` contains pre-trained networks that are 100% accurate for the following formal languages:

| Language | # Hidden units | Network checkpoint  | Test performance |
|----------|----------------|---------------------|------------------|  
| aⁿbⁿ     | 1              | an_bn.pt            | 100%             |
| aⁿbᵐcⁿ⁺ᵐ   | 1              | an_bm_c_n_plus_m.pt | 100%             | 
| Dyck-1   | 1              | dyck_1.pt           | 100%             | 

More on evaluation metrics [here](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00489/112499/Minimum-Description-Length-Recurrent-Neural).


## Fine-tuning

The networks use only standard activations (ReLU, tanh, sigmoid) and a final softmax layer, so they can be fine-tuned using regular backpropagation. See `examples` folder.

## Evolving networks from scratch for full MDL

MDLRNNs optimize the Minimum Description Length objective, which isn't differentiable. We use a genetic algorithm to evolve networks and then port them to PyTorch, freezing the architecture but not the weights. 

To evolve networks from scratch using a genetic algorithm, use the Genetic Algorithm MDLRNN trainer here:
https://github.com/taucompling/mdlrnn/

## TODO

- GPU support - although at the current size of the networks (1-2 hidden units), there's really no need.