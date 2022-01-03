from setuptools import setup, find_namespace_packages

# load the README file
with open(file = 'README.md', mode = 'r') as readme_handle:
    long_description = readme_handle.read()

# Setting up
setup(
    name= "vectory",
    version = '0.0.1',
    author = "ddamiana (Dave Amiana)",
    author_email = "<amiana.dave@gmail.com>",
    description = 'Dependency-Free Vector Algebra Library in Python.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/adeeconometrics/Vectory',
    packages = find_namespace_packages(
        where = 'vectory.vector.*'
    ),
    install_requires = [],
    keywords = ['python', 'vector', 'algebra', 'vector space', 'mathematics'],
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",

        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ]
)