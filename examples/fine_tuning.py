import torch
from torch import nn, optim

an_bn_net = torch.load("../models/an_bn.pt")

optimizer = optim.Adam(an_bn_net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

inputs = torch.Tensor([[0, 1, 1, 1, 2, 2, 2]])  # `#aaabbb`
targets = torch.Tensor([[1, 1, 1, 2, 2, 2, 0]])  # `aaabbb#`

inputs_one_hot = nn.functional.one_hot(inputs)
targets_one_hot = nn.functional.one_hot(targets)

nn.functional.one_hot()

if __name__ == "__main__":
    for epoch in range(1000):
        optimizer.zero_grad()
        output = an_bn_net(inputs, output_function="softmax")

        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} training loss: {loss.item():.3e}")
