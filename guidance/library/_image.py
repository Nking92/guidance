from .._guidance import guidance
from ..models._model_state import Image
import urllib
import typing
import http
import re


@guidance
def image(lm, src, allow_local=True):

    # load the image bytes
    # ...from a url
    if isinstance(src, str) and re.match(r"$[^:/]+://", src):
        name = src
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and isinstance(src, str):
        name = src
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from image file bytes
    elif isinstance(src, bytes):
        name = "bytes"
        bytes_data = src

    else:
        raise Exception(f"Unable to load image bytes from {src}!")

    # New - create image state
    # img = Image(name, bytes_data)
    # lm += img

    # old - set the image bytes
    bytes_id = str(id(bytes_data))
    lm = lm.set(bytes_id, bytes_data)
    lm += f"<|_image:{bytes_id}|>"
    return lm
