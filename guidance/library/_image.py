import guidance
import urllib
import typing
import http
import re

@guidance
def image(lm, src, allow_local=True):
    """
    Appends an image tag to the language model state with the source specified.

    This function handles loading image data from a URL, a local file path, or directly
    from image bytes. It then appends an image tag to the model state that references
    the image data. The function supports loading images from local paths if `allow_local`
    is set to True.

    Args:
        lm: The language model state to which the image tag will be appended.
        src (str or bytes): The source of the image. Can be a URL, a local file path,
                            or raw image bytes.
        allow_local (bool, optional): If True, allows loading images from local file paths.
                                    Defaults to True.

    Returns:
        The language model state with the appended image tag referencing the loaded image data.

    Raises:
        Exception: If the image data cannot be loaded from the specified source.

    Example:
        >>> lm = LanguageModelState()
        >>> lm = image(lm, 'https://example.com/image.png')
        # lm now includes an image tag with the source pointing to the provided URL.
    """

    # load the image bytes
    # ...from a url
    if isinstance(src, str) and re.match(r'$[^:/]+://', src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()
    
    # ...from a local path
    elif allow_local and isinstance(src, str):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from image file bytes
    elif isinstance(src, bytes):
        bytes_data = src
        
    else:
        raise Exception(f"Unable to load image bytes from {src}!")

    bytes_id = str(id(bytes_data))

    # set the image bytes
    lm = lm.set(bytes_id, bytes_data)
    lm += f'<|_image:{bytes_id}|>'
    return lm