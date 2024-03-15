import guidance
import os
import tiktoken
import re
from guidance import models, gen, any_char, any_char_but, regex, substring, substring_no_empty, with_temperature, system, user, assistant
from typing import Optional
import nbformat


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
            if include_exclude_check(file_name, include_file_regex, exclude_file_regex):
                matched_files.append(os.path.join(root, file_name))
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
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.ipynb'):
                file_contents[file_path] = extract_text_from_ipynb(file_path)
            else:
                file_contents[file_path] = f.read()
    return file_contents


def format_for_analysis(file_contents):
    """Format the contents for model prompting."""
    formatted_string = ""
    for file_path, content in file_contents.items():
        formatted_string += f"## File: {file_path}\n```{file_path.split('.')[-1]}\n{content}\n```\n\n"
    return formatted_string


# Orchestrator
def build_code_prompt(repo_tree_paths=None, repo_dir_paths=None, repo_file_paths=None, include_file_regex=None, exclude_file_regex=None):
    """Orchestrate the analysis of a repository."""
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
    prompt = f"""# Code Analysis
Please analyze the code provided below.

{formatted_code}"""
    return prompt


def code_chat(lm, code_prompt: str, system_prompt: Optional[str] = None, user_message: Optional[str] = None, **kwargs):
    kwargs.setdefault('temperature', 0.8)
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

    repo_tree_paths = ['/Users/nicholasking/code/ms/guidance/guidance']
    doc_paths = ['/Users/nicholasking/code/ms/guidance/README.md', '/Users/nicholasking/code/ms/guidance/notebooks/api_examples/models/OpenAI.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/tutorials/intro_to_guidance.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/anachronism.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/proverb.ipynb', '/Users/nicholasking/code/ms/guidance/notebooks/tutorials/chat.ipynb']
    match_file_regex = r'\.(py|cpp|ipynb|md)$'
    exclude_file_regex = r'(_consts.py)|(\.(pyc|so|dll)$)'

    code_prompt = build_code_prompt(repo_tree_paths=repo_tree_paths, repo_file_paths=doc_paths, include_file_regex=match_file_regex, exclude_file_regex=exclude_file_regex)
    # output to file for debugging
    # with open('code_prompt.md', 'w', encoding='utf-8') as f:
    #     f.write(code_prompt)
    code_chat(chatgpt4, code_prompt)