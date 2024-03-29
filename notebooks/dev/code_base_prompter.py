import guidance
import os
import tiktoken
import re
from guidance import models, gen, any_char, any_char_but, regex, substring, substring_no_empty, with_temperature, system, user, assistant
from typing import Optional
import nbformat
import logging
import json
from prompt_toolkit import prompt
from rich.markdown import Markdown
from rich.console import Console
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.lexers import PygmentsLexer


# Set up basic default logger with formatting
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def count_tokens(text: str):
    # Encoding for GPT-3 and later models
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    num_tokens = len(tokens)
    return num_tokens


def extract_text_from_ipynb(notebook_file):
    nb = nbformat.read(notebook_file, as_version=4)
    extracted_text = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            extracted_text += "```python\n" + cell['source'] + "\n```\n\n"
        elif cell['cell_type'] == 'markdown':
            extracted_text += "```ipynb\n" + cell['source'] + "\n```\n\n"
    return extracted_text


def gitignore_rules_to_regex(gitignore_str: str):
    rules = [rule.strip() for rule in gitignore_str.split('\n') if rule.strip() and not rule.startswith('#')]
    
    positive_patterns = []
    negative_patterns = []
    
    for rule in rules:
        if rule.startswith('!'):
            negative_patterns.append(rule[1:])
        else:
            positive_patterns.append(rule)
    
    def convert_pattern(pattern: str):
        if pattern.startswith('/'):
            pattern = '^' + pattern[1:]
        else:
            pattern = '(?:.*/)?' + pattern
        pattern = re.escape(pattern)
        pattern = pattern.replace(r'\*\*', '(?:.*/?)?')
        pattern = pattern.replace(r'\*', '[^/]*')
        pattern = pattern.replace(r'\?', '[^/]')
        pattern = pattern.replace(r'\[\^', '[^')
        return pattern
    
    positive_regex = '|'.join(convert_pattern(pattern) for pattern in positive_patterns)
    negative_regex = '|'.join(convert_pattern(pattern) for pattern in negative_patterns)
    
    if positive_regex and negative_regex:
        combined_regex = f'(?!(?:{negative_regex}))(?:{positive_regex})'
    elif positive_regex:
        combined_regex = positive_regex
    elif negative_regex:
        combined_regex = f'(?!(?:{negative_regex}))(?:.*)'
    else:
        combined_regex = '.*'  # Match all files by default
    
    return re.compile(combined_regex)


def include_exclude_check(file_name, include_file_regex=None, exclude_file_regex=None):
    includes = include_file_regex is None or re.search(include_file_regex, file_name) is not None
    excludes = exclude_file_regex is None or re.search(exclude_file_regex, file_name) is None
    # print("include_exclude_check", file_name, includes, excludes)
    return includes and excludes


def walk_and_match_files(start_path, include_file_regex=None, exclude_file_regex=None):
    """Walk through directories starting from start_path and collect files that match include_file_regex and don't match exclude_file_regex."""
    matched_files = []
    for root, _, files in os.walk(start_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if include_exclude_check(file_path, include_file_regex, exclude_file_regex):
                matched_files.append(file_path)
    return matched_files


def list_and_match_files(dir_path, include_file_regex=None, exclude_file_regex=None):
    """List files in dir_path and collect files that match include_file_regex and don't match exclude_file_regex."""
    matched_files = []
    for file_name in os.listdir(dir_path):
        if include_exclude_check(file_name, include_file_regex, exclude_file_regex):
            matched_files.append(os.path.join(dir_path, file_name))
    return matched_files


def read_files(file_paths):
    """Read the contents of the files."""
    file_contents = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.ipynb'):
                    file_contents[file_path] = extract_text_from_ipynb(file_path)
                else:
                    file_contents[file_path] = f.read()
        except UnicodeDecodeError:
            logging.error(f"Failed to decode file: {file_path}")
    return file_contents


def format_for_analysis(file_contents):
    """Format the contents for model prompting."""
    formatted_string = ""
    for file_path, content in file_contents.items():
        formatted_string += f"<file path={file_path}>\n{content}\n</file>\n\n"
    return formatted_string


def build_code_prompt(repo_tree_paths=None, repo_dir_paths=None, repo_file_paths=None, include_file_regex=None, exclude_file_regex=None):
    """Build a large string containing all the text in the repo files."""
    all_file_paths = []
    if repo_tree_paths is not None:
        for start_path in repo_tree_paths:
            all_file_paths.extend(walk_and_match_files(start_path, include_file_regex, exclude_file_regex))

    if repo_dir_paths is not None:
        for dir_path in repo_dir_paths:
            all_file_paths.extend(list_and_match_files(dir_path, include_file_regex, exclude_file_regex))

    if repo_file_paths is not None:
        all_file_paths.extend(repo_file_paths)
    all_file_contents = read_files(all_file_paths)
    formatted_code = format_for_analysis(all_file_contents)
    return formatted_code


def generate_gitignore(lm, stack_description: str, **kwargs) -> str:
    kwargs.setdefault('temperature', 0.3)
    kwargs.setdefault('max_tokens', 4000)

    if not isinstance(lm, models.Chat):
        raise Exception("Only chat models supported.")

    system_prompt = f"You are a helpful assistant that generates highly detailed .gitignore files based on a given stack description. Respond with the generated .gitignore file and nothing more. Do not include Markdown formatting."

    with system():
        lm += system_prompt

    with user():
        lm += f"Generate for the following stack:\n\n<stack>{stack_description}</stack>"

    with assistant():
        lm += gen(name="gitignore", **kwargs)

    return lm['gitignore']


@guidance
def setup_code_chat(lm, code_prompt: str, system_prompt: Optional[str] = None):
    if not isinstance(lm, models.Chat):
        raise Exception("Only chat models supported.")

    if system_prompt is None:
        system_prompt = "You are a friendly, helpful software engineering assistant. Provide insights into code quality, potential issues, and suggestions for improvement. Answer the user's questions. Format your responses in Markdown."

    with system():
        lm += system_prompt

    with user():
        # Reiterate the system prompt to ensure the assistant pays attention to it
        lm += f"<code_base>{code_prompt}</code_base>\n\n<instructions>{system_prompt}</instructions>"
    
    with assistant():
        lm += "Thank you for providing your code. How can I assist you today?"

    return lm


@guidance
def chat_with_model(lm, **kwargs):
    if not isinstance(lm, models.Chat):
        raise Exception("Only chat models supported.")

    temperature = kwargs.get('temperature', lm.get('temperature', 0.8))
    max_tokens = kwargs.get('max_tokens', lm.get('max_tokens', 1000))
    console = Console()

    print("=== Beginning of chat. Press escape then enter to send your message. Type 'exit' or 'quit' to end the chat. ===")
    while True:
        print("~*~ You ~*~")
        user_message = prompt(lexer=PygmentsLexer(MarkdownLexer), multiline=True)
        cln = user_message.strip().lower()
        if cln == "exit" or cln == "quit":
            break

        with user():
            lm += user_message

        with assistant():
            lm += gen(**kwargs, name="chat_response", temperature=temperature, max_tokens=max_tokens)

        markdown_response = Markdown(lm["chat_response"])
        print("\n~*~ Assistant ~*~")
        console.print(markdown_response)
    
    return lm


def save_conversation_to_file(lm, file_path, metadata={}):
    state_dict = {
        'lm': str(lm),
        'metadata': metadata
    }
    json.dump(state_dict, open(file_path, 'w', encoding='utf-8'))


@guidance
def load_conversation_from_file(lm, file_path):
    """ loads conversation into lm prompt and adds metadata to lm attributes """
    state_dict = json.load(open(file_path, 'r', encoding='utf-8'))
    lm += state_dict['lm']
    for key, value in state_dict['metadata'].items():
        lm.set(key, value)
    return lm


def generate_conversation_name(lm) -> str:
    """Generates a name for the conversation by prompting the model."""
    with system():
        lm += "You are a conversation naming assistant. You are given a series of messages between a user and another assistant. Directly output a name for this conversation."
    
    with user():
        lm += "Name this conversation. Directly print the name with nothing extra."
    
    with assistant():
        lm += gen(name="conversation_name", max_tokens=30, temperature=0.8)

    return lm['conversation_name']


def chat_demo(lm):
    # setup to prompt guidance repo
    # repo_tree_paths = ['/Users/nicholasking/code/ms/guidance/guidance']
    # doc_paths = ['/Users/nicholasking/code/ms/guidance/README.md', '/Users/nicholasking/code/ms/guidance/notebooks/api_examples/models/OpenAI.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/tutorials/intro_to_guidance.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/anachronism.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/proverb.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/tutorials/chat.ipynb']
    # code_prompt = build_code_prompt(repo_tree_paths=repo_tree_paths, repo_file_paths=doc_paths, include_file_regex=match_file_regex, exclude_file_regex=exclude_file_regex)
    # match_file_regex = r'\.(py|cpp|ipynb|md)$'

    exclude_file_regex = r'(_consts.py)|(\.(pyc|so|dll)$)|(node_modules)|(env)|(static)|(package-lock.json)'
    repo_tree_paths = ['/Users/nicholasking/code/ms/guidance/laser-rdk/guidance-evaluation/code-chat']

    code_prompt = build_code_prompt(repo_tree_paths=repo_tree_paths, exclude_file_regex=exclude_file_regex)
    full_prompt = f"Analyze the code below.\n\n{code_prompt}"
    print(f"Code prompt has {count_tokens(full_prompt)} tokens.")

    code_chat_lm = lm + setup_code_chat(full_prompt)

    code_chat_lm += chat_with_model()

    conv_name_input = input("Input a conversation name, or press Enter to generate a name: ")
    if len(conv_name_input) > 0:
        conversation_name = conv_name_input
    else:
        conversation_name = generate_conversation_name(code_chat_lm)
    
    conversation_name = conversation_name.replace(' ', '_')
    file_name = f'chat_{conversation_name}.json'
    print(f"Saving conversation to file {file_name}")

    save_conversation_to_file(code_chat_lm, file_name)


def gitignore_generator_demo(lm):
    stack = "A full stack app using React, Flask, protobuf, and PostgreSQL. There is also an Android app that uses Java and Kotlin."
    gitignore = generate_gitignore(lm, stack)
    print(gitignore)
    regex = gitignore_rules_to_regex(gitignore)
    print(regex.pattern)

if __name__ == '__main__':
    azure_model = os.getenv("AZUREAI_CHAT_MODEL", "Please set the model")
    azure_endpoint = os.getenv("AZUREAI_CHAT_ENDPOINT", "Please set the endpoint")
    azure_api_key=os.getenv("AZUREAI_CHAT_KEY", "Please set API key")

    lm = models.AzureOpenAI(
        model=azure_model,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        echo=False
    )
    chat_demo(lm)