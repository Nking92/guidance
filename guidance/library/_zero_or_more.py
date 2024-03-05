import guidance
from .._grammar import select

@guidance(stateless=True)
def zero_or_more(model, value):
    """
    Creates a grammar function that matches zero or more occurrences of a given pattern,
    equivalent to the Kleene star operator (*) in regular expressions.

    Args:
        model: The language model state to which the zero-or-more grammar function
            will be appended.
        value: The pattern to match zero or more times. Can be a string or another
            grammar function.

    Returns:
        The language model state with the appended grammar function that matches
        zero or more occurrences of the specified pattern.

    Example:
        >>> lm += zero_or_more(lm, 'word ')
        # lm now includes a grammar function that can match the pattern 'word ' zero or more times.
    """
    return model + select(["", value], recurse=True)