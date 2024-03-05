import guidance
from .._grammar import byte_range

@guidance(stateless=True)
def any_char(lm):
     """
     Generates a grammar function that matches any single byte character.

     This function extends the given language model (`lm`) by appending a grammar
     function that allows for any single byte character in the range from 0x00 to 0xff,
     which includes all ASCII characters.
     """
    # TODO: extend this to support utf-8 encoded multibyte unicode characters
    return lm + byte_range(b'\x00', b'\xff')