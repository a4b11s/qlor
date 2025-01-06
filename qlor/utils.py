import shutil


def print_into_middle_of_terminal(text: str):
    terminal_width = shutil.get_terminal_size().columns
    print(
        f"{"*" * ((terminal_width - len(text) - 1) // 2)} {text} {"*" * ((terminal_width - len(text) - 2) // 2)}"
    )
