from .._grammar import select
from ._char_range import char_range

def char_set(def_string: str):
    """
    Creates a grammar function that matches any single character from a defined set.

    Args:
        def_string (str): A string defining the set of characters to match.
                        It can include ranges (e.g., 'a-z') and escaped characters (e.g., '\\n').

    Returns:
        A grammar function that matches any single character from the specified set.

    Raises:
        Exception: If the range definition in `def_string` includes multibyte characters,
        which are not supported.

    Example:
        >>> char_set('a-zA-Z0-9')
        # This will return a grammar function that matches any alphanumeric character.

        >>> char_set('a-dx-z')
        # This will return a grammar function that matches characters a to d and x to z.

        >>> char_set('\\n\\t')
        # This will return a grammar function that matches newline and tab characters.
    """
    parts = []
    pos = 0
    while pos < len(def_string):
        if pos + 2 < len(def_string) and def_string[pos + 1] == "-":
            parts.append(char_range(def_string[pos], def_string[pos + 2]))
            pos += 3
        elif pos + 1 < len(def_string) and def_string[pos] == "\\":
            parts.append(def_string[pos + 1])
            pos += 2
        else:
            parts.append(def_string[pos])
            pos += 1
    return select(parts)