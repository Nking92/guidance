import traceback
from guidance import assistant, gen, models, system, user
import os

def test_gemini_pro():

    try:
        vmodel = models.GoogleAI("models/gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"), echo=False)
    except Exception as e:
        traceback.print_exc()

    lm = vmodel

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=100, temperature=0.8)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=100, temperature=0.8)

    # second time to make sure cache reuse is okay
    print(lm)
    # lm = vmodel

    # with user():
    #     lm += "The economy is crashing!"

    # with assistant():
    #     lm += gen("test1", max_tokens=100, temperature=0.8)

    # with user():
    #     lm += "What is the best again?"

    # with assistant():
    #     lm += gen("test2", max_tokens=100, temperature=0.8)

    # print(lm)

def test_gemini_pro2():

    try:
        vmodel = models.GoogleAI("models/gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"), echo=False)
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

test_gemini_pro2()
