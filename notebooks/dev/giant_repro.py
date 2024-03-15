import guidance
import os
import tiktoken
import re
from guidance import models, gen, any_char, any_char_but, regex, substring, substring_no_empty, with_temperature, system, user, assistant
from typing import Optional
import nbformat
from datetime import datetime
from time import sleep


def count_tokens(text: str):
    # Encoding for GPT-3 and later models
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    num_tokens = len(tokens)
    return num_tokens


def send_chat(lm, code_prompt: str, system_prompt: Optional[str] = None, user_message: Optional[str] = None, **kwargs):
    kwargs.setdefault('temperature', 0.01)
    kwargs.setdefault('max_tokens', 1000)

    if not isinstance(lm, models.Chat):
        raise Exception("Only chat models supported.")

    if system_prompt is None:
        system_prompt = "Act as an expert software architect. Provide insights into code quality, potential issues, and suggestions for improvement. Answer the user's questions. Format your responses in Markdown."

    with system():
        lm += system_prompt

    with user():
        lm += code_prompt
        if user_message is not None:
            lm += f"\n# Initial User Message\n{user_message}"

    with assistant():
        lm += gen(name="response", **kwargs)

    print(lm['response'])


if __name__ == '__main__':
    # mistral = models.LlamaCpp("/Users/nicholasking/code/models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf", n_gpu_layers=-1, n_ctx=4096)

    azure_model = os.getenv("AZUREAI_CHAT_MODEL", "Please set the model")
    azure_endpoint = os.getenv("AZUREAI_CHAT_ENDPOINT", "Please set the endpoint")
    azure_api_key=os.getenv("AZUREAI_CHAT_KEY", "Please set API key")

    chatgpt4 = models.AzureOpenAI(
        model=azure_model,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        echo=False
    )

    # Hello world X 5000 = 10k tokens
    prompt = "Hello world" * 5000 * 6
    while True:
        try:
            print("Sending prompt with {} tokens".format(count_tokens(prompt)))
            send_chat(chatgpt4, prompt)
            print("-------------------")
            # thousand_chars = "a b c d: ?e" * 1000
            prompt = prompt + ("Hello world" * 2500)
            # sleep for a bit to avoid rate limiting
            sleep_time = 3 * count_tokens(prompt) / 1000
            print("Sleeping for {} seconds".format(sleep_time))
            sleep(sleep_time)
        except Exception as e:
            print("Error: {}".format(e))
            print("Saving prompt and error to file")
            with open("giant_repro_out.txt", "a") as f:
                # Write the timestamp
                f.write("Timestamp: ")
                f.write(str(datetime.now()))
                f.write("\nPrompt: ")
                f.write(prompt)
                f.write("\n")
                f.write(str(e))
                f.write("\n")
                f.write(repr(e))
                f.write("-"*30)
                f.write("\n\n")
            print("File saved, waiting 60 seconds before retrying.")
            sleep(60)
            continue