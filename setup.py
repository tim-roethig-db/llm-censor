from setuptools import setup, find_packages


setup(
    name="llm_censor",
    version="0.0.1",
    author="Tim Röthig",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "pandas",
        "torch",
        "pyreft"
    ]
)
