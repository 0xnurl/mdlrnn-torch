import itertools

import torch
from torch import nn

ComputationGraph = dict[int, tuple[tuple[int, nn.Linear], ...]]


class MDLRNN(nn.Module):
    def __init__(
        self,
        computation_graph: ComputationGraph,
        layer_to_memory_weights: dict[int, nn.Linear],
        memory_to_layer_weights: dict[int, nn.Linear],
        layer_to_activation_to_units: dict[int, dict[int, tuple[int, ...]]],
    ):
        super(MDLRNN, self).__init__()
        self._computation_graph = computation_graph
        self._layer_to_activation_to_units = layer_to_activation_to_units
        self._memory_to_layer_weights = memory_to_layer_weights
        self.layer_to_memory_weights = layer_to_memory_weights

        self._memory_size = memory_to_layer_weights[
            min(memory_to_layer_weights)
        ].in_features

        self.module_list = nn.ModuleList(
            [x[1] for x in list(itertools.chain(*self._computation_graph.values()))]
            + list(self.layer_to_memory_weights.values())
            + list(self._memory_to_layer_weights.values())
        )

    def forward(self, inputs, memory=None, output_layer=None):
        """
        :param inputs: batched tensor of shape `(batch_size, sequence_length, num_input_classes)`.
        :param memory: batch tensor of shape `(batch_size, memory_size)`.
        :param output_layer: function to apply to outputs: `None` for raw logits, `"softmax"`, or `"normalize"` for simple normalization.
        :return: tensor of shape `(batch_size, sequence_length, num_output_classes)`.
        """

        def recurrence(inputs_inner, memory_inner):
            input_layer_num = min(self._computation_graph)
            layer_to_vals = {input_layer_num: inputs_inner}

            memory_out = torch.zeros(
                (
                    inputs_inner.shape[0],
                    self._memory_size,
                )
            )

            source_layers = set(self._computation_graph.keys()) | set(
                self.layer_to_memory_weights.keys()
            )
            for source_layer in sorted(source_layers):
                # Add memory.
                if source_layer in self._memory_to_layer_weights:
                    memory_weights = self._memory_to_layer_weights[source_layer]
                    incoming_memory = memory_weights(memory_inner)
                    layer_to_vals[source_layer] = (
                        layer_to_vals[source_layer] + incoming_memory
                    )

                # Apply activations.
                source_layer_activations_to_unit = self._layer_to_activation_to_units[
                    source_layer
                ]
                activation_vals = self._apply_activations(
                    source_layer_activations_to_unit, layer_to_vals[source_layer]
                )
                layer_to_vals[source_layer] = activation_vals

                # Feed-forward.
                for target_layer, current_weights in self._computation_graph.get(
                    source_layer, ()
                ):
                    source_layer_val = layer_to_vals[source_layer]
                    target_layer_val = current_weights(source_layer_val)

                    if target_layer in layer_to_vals:
                        layer_to_vals[target_layer] = (
                            layer_to_vals[target_layer] + target_layer_val
                        )
                    else:
                        layer_to_vals[target_layer] = target_layer_val

                # Write to memory.
                if source_layer in self.layer_to_memory_weights:
                    memory_out = memory_out + self.layer_to_memory_weights[
                        source_layer
                    ](layer_to_vals[source_layer])

            # Add memory and apply activations to output layer.
            output_layer_num = max(layer_to_vals)
            y_out = layer_to_vals[output_layer_num]

            if output_layer in self._memory_to_layer_weights:
                memory_to_output_layer = self._memory_to_layer_weights[
                    output_layer_num
                ](memory_inner)
                y_out = y_out + memory_to_output_layer

            y_out = self._apply_activations(
                self._layer_to_activation_to_units[output_layer_num], y_out
            )

            return y_out, memory_out

        if memory is None:
            memory = torch.zeros(
                (inputs.shape[0], self._memory_size),
                dtype=next(iter(self._computation_graph.values()))[0][1].weight.dtype,
            )

        outputs = []
        for step in range(inputs.shape[1]):
            y_t, memory = recurrence(inputs[:, step], memory)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)

        if output_layer is not None:
            if output_layer == "softmax":
                outputs = torch.softmax(outputs, dim=-1)
            elif output_layer == "normalize":
                outputs = torch.clamp(outputs, min=0, max=None)
                outputs = nn.functional.normalize(outputs, p=1, dim=-1)
            else:
                raise ValueError(output_layer)

        return outputs, memory

    @staticmethod
    def _apply_activations(activation_to_unit, layer_vals) -> torch.Tensor:
        for activation_id in activation_to_unit:
            activation_unit_idxs = activation_to_unit[activation_id]
            if activation_id == 0:  # Identity.
                continue
            activation_func = {
                1: torch.relu,
                2: torch.sigmoid,
                3: torch.tanh,
                4: torch.square,
            }[activation_id]
            layer_vals[:, activation_unit_idxs] = activation_func(
                layer_vals[:, activation_unit_idxs]
            )
        return layer_vals
