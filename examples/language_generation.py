import torch

input_strings = [
    "#ab",
    "#aab",
    "#aaab",
    "#aaaab",
    "#aaaaab",
    "#aaaaaab",
    "#aaaaaaab",
    "#aaaaaaaab",
    "#aaaaaaaaab",
    "#aaaaaaaaaab",
    "#aaaaaaaaaaab",
    "#aaaaaaaaaaaab",
    "#aaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaaaaaab",
    "#aaaaaaaaaaaaaaaaaaaaab",
]

vocab = {"#": 0, "a": 1, "b": 2}
vocab_idxs = {0: "#", 1: "a", 2: "b"}

an_bn_net = torch.load("../models/an_bn.pt")

for input_string in input_strings:
    inputs_one_hot = torch.zeros((1, len(input_string), len(vocab)))
    for i, s in enumerate(input_string):
        inputs_one_hot[0, i, vocab[s]] = 1

    memory = None
    generated_string = ""

    while True:
        output, memory = an_bn_net(inputs_one_hot, memory, output_layer="softmax")

        predicted_next_class = output[0, -1].argmax().item()
        generated_string += vocab_idxs[predicted_next_class]

        if vocab_idxs[predicted_next_class] == "#":
            break

        inputs_one_hot = torch.zeros((1, 1, len(vocab)))
        inputs_one_hot[0, 0, predicted_next_class] = 1

    print(f"{input_string} ==> {generated_string}")
