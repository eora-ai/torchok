from typing import Union, Optional


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
        """Round number of filters based on depth multiplier."""
        if not multiplier:
            return channels
        channels *= multiplier
        return make_divisible(channels, divisor, channel_min)

def make_divisible(value: Union[int, float], divisor: int = 8, min_value: Optional[int] = None):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be divisible by the divisor.

    Args:
        value: The original channel number.
        divisor: The divisor to fully divide the channel number.
        min_value: The minimum value of the output channel.
    """
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value