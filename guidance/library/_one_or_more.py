import guidance
from .._grammar import select

@guidance(stateless=True)
def one_or_more(model, value):
    """
    Creates a grammar function that matches one or more occurrences of a given pattern.

    Args:
        model: The language model state to which the one-or-more grammar function
            will be appended.
        value: The pattern to match one or more times. Can be a string or another
            grammar function.

    Returns:
        The language model state with the appended grammar function that matches
        one or more occurrences of the specified pattern.

    Example:
        >>> pattern = 'word'
        >>> lm += one_or_more(pattern)
        # lm now includes a grammar function for one or more occurrences of 'word'.
    """
    return model + select([value], recurse=True)