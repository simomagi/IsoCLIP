from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()


def is_image_file(filename: str) -> bool:
    """
    Checks whether a file has an image extension.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file has an image extension, otherwise False.
    """
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
