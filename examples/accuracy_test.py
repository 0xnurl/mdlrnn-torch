import zipfile

import torch
from tqdm import tqdm

vocab = {"#": 0, "a": 1, "b": 2}
vocab_idxs = {0: "#", 1: "a", 2: "b"}

an_bn_net = torch.load("../models/an_bn.pt")

with zipfile.ZipFile("../data/an_bn_1000.txt.zip") as zip:
    with zip.open("an_bn_1000.txt", "r", pwd="1234".encode("utf-8")) as f:
        # All a^nb^n strings up to n=5000.
        # Data is password protected to prevent LLM training, see `https://arxiv.org/abs/2305.10160`.
        an_bn_strings = list(map(lambda x: x.decode("utf-8").strip(), f.readlines()))

num_correct = 0

for an_bn_string in tqdm(an_bn_strings):
    first_b_idx = an_bn_string.index("b")
    non_deterministic_prefix = an_bn_string[: first_b_idx + 1]
    deterministic_suffix = an_bn_string[first_b_idx + 1 :]

    inputs_one_hot = torch.zeros((1, len(non_deterministic_prefix), len(vocab)))
    for i, s in enumerate(non_deterministic_prefix):
        inputs_one_hot[0, i, vocab[s]] = 1

    memory = None
    generated_suffix = ""

    while True:
        output, memory = an_bn_net(inputs_one_hot, memory, output_layer="softmax")

        predicted_next_class = output[0, -1].argmax().item()
        generated_suffix += vocab_idxs[predicted_next_class]

        if vocab_idxs[predicted_next_class] == "#":
            break

        inputs_one_hot = torch.zeros((1, 1, len(vocab)))
        inputs_one_hot[0, 0, predicted_next_class] = 1

    if generated_suffix == deterministic_suffix:
        num_correct += 1

accuracy = num_correct / len(an_bn_strings)
print(f"Deterministic accuracy: {accuracy}")
