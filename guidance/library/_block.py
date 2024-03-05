from guidance import models

class ContextBlock:
    """
    A context manager that defines a block of text with an opener and a closer.

    This context manager is used to create structured blocks of text in a guidance
    program, with specified opening and closing strings. It also optionally associates
    a name with the block for reference within the program.

    Attributes:
        opener (str): The opening string for the block.
        closer (str): The closing string for the block.
        name (str, optional): An optional name to associate with the block.

    Example:
        >>> with ContextBlock("BEGIN ", " END", name="example"):
        ...     # Generation logic within the block
        ...     pass
    """
    def __init__(self, opener, closer, name=None):
        self.opener = opener
        self.closer = closer
        self.name = name

    def __enter__(self):
        """Registers the block as open in the global model state upon entry."""
        models.Model.open_blocks[self] = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Removes the block from the global model state upon exit."""
        del models.Model.open_blocks[self]

def block(name=None, opener="", closer=""):
    """
    Factory function to create a `ContextBlock` with the given parameters.

    Args:
        name (str, optional): A name to associate with the block.
        opener (str): The string to append to the model state at the start of the block.
        closer (str): The string to append to the model state at the end of the block.

    Returns:
        ContextBlock: An instance of the `ContextBlock` context manager.

    Example:
        >>> with block(name="example", opener="BEGIN ", closer=" END"):
        ...     # Generation logic within the block
        ...     pass
    """
    return ContextBlock(opener, closer, name=name)