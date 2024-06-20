import traceback
from guidance import assistant, gen, models, system, user
import os

def test_gpt4o():

    try:
        vmodel = models.OpenAI("gpt-4o", echo=False)
    except Exception as e:
        traceback.print_exc()

    lm = vmodel

    with user():
        lm += "What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak?"
        lm += "What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak?"
        lm += "What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak?"
        lm += "What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak?"
        lm += "What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak? What's the best way to learn how to cook a steak?"

    with assistant():
        lm += gen("test1", max_tokens=5, temperature=0.8)

    print(lm)

test_gpt4o()
