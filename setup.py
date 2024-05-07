from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements file and returns a list of requirements.
    """
    requirements = []
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Bor",
    author_email="borc23@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
