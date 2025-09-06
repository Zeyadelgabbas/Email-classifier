from setuptools import setup , find_packages

def get_requirments(file_path:str):
    
    HYPHEN_DOT_E = '-e .'
    with open(file_path) as file:
        requirments= [line.rstrip("\n") for line in file.readlines()] 

        if HYPHEN_DOT_E in requirments:
            requirments.remove(HYPHEN_DOT_E)
    return requirments


setup(
name = 'EmailClassifier',
version='0.0.1',
author='Zeyad',
author_email='Zeyadelgabas@gmail.com',
packages=find_packages(),
install_requires = get_requirments('requirments.txt')

)