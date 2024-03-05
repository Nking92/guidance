from .._grammar import byte_range
    
def char_range(low: str, high: str):
    """
    Creates a grammar function that matches any single character within a specified range.

    This function takes two single-character strings representing the lower and upper bounds
    of a character range, and returns a grammar function that matches any character within
    that range, inclusive. Only single-byte character ranges are currently supported.

    Args:
        low (str): A single-byte string representing the low end of the character range.
        high (str): A single-byte string representing the high end of the character range.

    Returns:
        A grammar function that matches any single character within the specified range.

    Raises:
        Exception: If either `low` or `high` is a multibyte character.

    Example:
        >>> char_range('a', 'z')
        # This will return a grammar function that matches any lowercase letter from a to z.
    """
    low_bytes = bytes(low, encoding="utf8")
    high_bytes = bytes(high, encoding="utf8")
    if len(low_bytes) > 1 or len(high_bytes) > 1:
        raise Exception("We don't yet support multi-byte character ranges!")
    return byte_range(low_bytes, high_bytes)