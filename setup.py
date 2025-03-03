from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='timekan',
    version='0.1.2',
    description='Python library designed to integrate Kolmogorov Arnold Networks with recurrent mechanisms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Samer Makni',
    author_email='samermakni@outlook.com',
    url='https://github.com/samermakni/timekan', 
    packages=find_packages(),
    install_requires=[ 
        "torch>=2.4.0",
        "numpy>=1.20.0",
        "cuda_selector>=0.1.0"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
