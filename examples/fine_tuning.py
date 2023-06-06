import torch
from torch import nn, optim

vocab = {"#": 0, "a": 1, "b": 2}
vocab_idxs = {0: "#", 1: "a", 2: "b"}


training_string = "#aaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbb"
target_string = "aaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbb#"

inputs_one_hot = torch.zeros((1, len(training_string), len(vocab)))
targets_one_hot = torch.zeros((1, len(training_string), len(vocab)))

for i in range(len(training_string)):
    inputs_one_hot[0, i, vocab[training_string[i]]] = 1
    targets_one_hot[0, i, vocab[target_string[i]]] = 1

an_bn_net = torch.load("../models/an_bn.pt")
optimizer = optim.Adam(an_bn_net.parameters(), lr=0.001)
cross_entropy_loss = nn.CrossEntropyLoss()

for epoch in range(1000):
    optimizer.zero_grad()
    output, _ = an_bn_net(inputs_one_hot, output_layer=None)  # Raw logits.

    loss = cross_entropy_loss(output, targets_one_hot)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} training loss: {loss.item():.3e}")
