import guidance
from .._grammar import select, string

@guidance(stateless=True, dedent=False)
def prefix_tree(lm, strings, partial_matches=False):
    """
    Creates a grammar function that represents a prefix tree (trie) for efficient matching.

    This function constructs a trie that can match any of the strings provided in
    the `strings` list, based on their common prefixes. It's useful for efficiently matching
    against a large list of potential options where many share common starting characters.

    Args:
        lm: The language model state to which the prefix tree grammar will be appended.
        strings (list of str): A list of strings to include in the prefix tree.
        partial_matches (bool, optional): If True, allows the grammar to match partial strings
                                          that begin with any of the prefixes. Defaults to False.

    Returns:
        The language model state with the appended grammar function representing the prefix tree.

    Example:
        >>> strings = ["apple", "apricot", "banana", "berry", "blueberry"]
        >>> lm += prefix_tree( strings)
        # lm now includes a grammar function that can match any of the strings in the list.
    """

    if len(strings) == 0:
        return lm

    # group the strings by their starting character
    char_groups = {}
    for s in strings:
        if len(s) > 0:
            if s[0] not in char_groups:
                char_groups[s[0]] = []
            char_groups[s[0]].append(s[1:])
    
    # enable any empty followup if partial matches are allowed
    if partial_matches:
        char_groups[""] = []
    
    # recursively build the tree
    suboptions = [string(k) + prefix_tree(v, partial_matches=partial_matches) for k,v in char_groups.items()]

    return lm + select(suboptions, skip_checks=True) # we skip normal type checks for speed