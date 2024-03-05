import guidance
from .._grammar import select

@guidance(stateless=True)
def optional(lm, value):
    """
    Creates a grammar function that matches an optional element in a pattern.
    This is functionally equivalent to select(["", value]).

    Args:
        lm: The language model state to which the optional grammar function
            will be appended.
        value: The pattern that is optional in the generated text. Can be a string
            or another grammar function.

    Returns:
        The language model state with the appended grammar function that matches
        an optional occurrence of the specified pattern.

    Example:
        >>> lm += optional("perhaps ")
        # lm now includes a grammar function for the optional word "perhaps".
    """
    return lm + select([value, ""])