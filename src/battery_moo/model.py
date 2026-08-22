"""Architecture bookkeeping independent of PyTorch."""

def lstm_parameter_count(input_size: int, hidden: int, layers: int, output_size: int = 2) -> int:
    """Count the exact PyTorch LSTM plus 128/64 regression-head parameters."""
    recurrent = 0
    for layer in range(layers):
        layer_input = input_size if layer == 0 else hidden
        recurrent += 4 * hidden * (layer_input + hidden) + 8 * hidden
    head = hidden * 128 + 128 + 128 * 64 + 64 + 64 * output_size + output_size
    return recurrent + head

