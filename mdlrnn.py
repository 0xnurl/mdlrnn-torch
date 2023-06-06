import torch
from torch import nn


class MDLRNN(nn.Module):
    _ACTIVATION_ID_TO_TORCH = {
        0: None,  # identity
        1: torch.relu,
        3: torch.tanh,
        4: torch.square,
        2: torch.sigmoid,
    }

    def __init__(
        self,
        computation_graph_weights: tuple[tuple[int, int, nn.Linear]],
        memory_to_layer_weights: dict[int, nn.Linear],
        memory_unit_idxs_per_layer: dict[int, list[int]],
        layer_to_activation_to_units: dict[int, dict[int, frozenset[int]]],
    ):
        super(MDLRNN, self).__init__()
        self._computation_graph_weights = computation_graph_weights
        self._layer_to_activation_to_units = layer_to_activation_to_units
        self._memory_to_layer_weights = memory_to_layer_weights
        self._memory_unit_idxs_per_layer = memory_unit_idxs_per_layer

        self._memory_size = memory_to_layer_weights[
            min(memory_to_layer_weights)
        ].in_features

        self.module_list = nn.ModuleList(
            [x[2] for x in self._computation_graph_weights]
            + list(self._memory_to_layer_weights.values())
        )

    def forward(self, inputs, memory=None):
        def recurrence(inputs_inner, memory_inner):
            first_layer_num = self._computation_graph_weights[0][0]
            layer_to_vals = {first_layer_num: inputs_inner}

            for i, (source_layer, target_layer, current_weights) in enumerate(
                self._computation_graph_weights
            ):
                if i < len(self._computation_graph_weights) - 1:
                    next_target_layer = self._computation_graph_weights[i + 1][1]
                else:
                    next_target_layer = None

                source_layer_val = layer_to_vals[source_layer]
                target_layer_val = current_weights(source_layer_val)

                if target_layer in layer_to_vals:
                    layer_to_vals[target_layer] = (
                        layer_to_vals[target_layer] + target_layer_val
                    )
                else:
                    layer_to_vals[target_layer] = target_layer_val

                if next_target_layer != target_layer:
                    # Only add memory and apply activations when all inputs to layer have been added.

                    # Add memory to layer.
                    target_memory_weights = self._memory_to_layer_weights[target_layer]
                    incoming_memory = target_memory_weights(memory_inner)
                    layer_to_vals[target_layer] = (
                        layer_to_vals[target_layer] + incoming_memory
                    )
                    # Apply activations.
                    target_layer_activations_to_unit = (
                        self._layer_to_activation_to_units[target_layer]
                    )
                    activation_vals = self._apply_activations(
                        target_layer_activations_to_unit, layer_to_vals[target_layer]
                    )
                    layer_to_vals[target_layer] = activation_vals

            memory_out = torch.zeros((inputs.shape[0], 0))

            for layer_num in sorted(self._memory_unit_idxs_per_layer):
                layer_mem_idxs = self._memory_unit_idxs_per_layer[layer_num]
                if len(layer_mem_idxs) == 0:
                    continue

                layer_val = layer_to_vals[layer_num]
                memory_from_layer = layer_val[:, layer_mem_idxs]
                memory_out = torch.concat([memory_out, memory_from_layer], dim=1)

            y_out = layer_to_vals[max(layer_to_vals)]
            return y_out, memory_out

        if memory is None:
            memory = torch.zeros(
                (
                    inputs.shape[0],
                    self._memory_size,
                )
            )

        outputs = []
        for step in range(inputs.shape[1]):
            y_t, memory = recurrence(inputs[:, step], memory)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    @classmethod
    def _apply_activations(cls, activation_to_unit, layer_vals) -> torch.Tensor:
        for activation_id in activation_to_unit:
            activation_unit_idxs = activation_to_unit[activation_id]
            if activation_id == 0:  # Identity.
                continue
            activation_func = cls._ACTIVATION_ID_TO_TORCH[activation_id]
            layer_vals[:, activation_unit_idxs] = activation_func(
                layer_vals[:, activation_unit_idxs]
            )
        return layer_vals
