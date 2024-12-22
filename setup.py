from setuptools import find_packages, setup



# Read README.md for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

#Read requirement.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name= "mlops_modular_project",
    version= "0.1.0",
    author = "Azim",
    author_email = "khanzim.eee@gmail.com",
    description= "A modular MLops pipeline project",
    long_description= long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/zim10/Mlops_project",
    packages= find_packages(),
    classifier =[
        "Development Status :: 3 - Beta",
        "Intendent audience :: ML Engineer",
        "Programming language :: Python >=3.8",
        "Operating systems :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires= required,
    extras_require={
        'dev': [
            'pytest>=7.1.1',
            'pytest-cov>=2.12.1',
            'flake8>=3.9.0',
            'black>=22.3.0',
            'isort>=5.10.1',
            'mypy>=0.942',
        ],
    }   
)
