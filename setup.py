from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirement(file_path: str) -> List[str]:
    """This function returns the list of requirements"""
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="MLproject",
    author="princepalsingh11",
    author_email="krishraj332211@gmail.com",
    version="0.0.1",
    packages=find_packages(),   
    install_requires=get_requirement("requirements.txt")  
)