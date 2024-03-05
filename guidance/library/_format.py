from ._block import block

def monospace():
    """
    Creates a context block that applies monospaced font styling to enclosed text.

    This function generates a context block for use within a guidance program to apply
    monospaced font styling, commonly used for code or preformatted text. The block
    is defined with HTML `span` tags that include inline CSS to set the font family
    to a monospaced font and the font size to 13 pixels.

    Returns:
        ContextBlock: An instance of the `ContextBlock` context manager configured for
                    monospaced styling.

    Example:
        >>> with monospace():
        ...     # Text generated within this block will be styled in monospace font.
        ...     pass
    """
    return block(opener="<||_html:<span style='font-family: Menlo, Monaco, monospace; font-size: 13px;'>_||>", closer="<||_html:</span>_||>")

