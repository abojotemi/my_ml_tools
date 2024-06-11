# setup.py

from setuptools import setup, find_packages

setup(
    name='my_ml_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'ipywidgets'
    ],
    author='Ab-ojo Abraham',
    author_email='abojotemi@gmail.com',
    description='A package for plotting decision boundaries in machine learning',
    url='https://github.com/abojotemi/my_ml_tools',
)




