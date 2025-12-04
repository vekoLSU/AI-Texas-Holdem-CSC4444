from setuptools import setup, find_packages

setup(
    name="poker-bot",
    version="1.0.0",
    description="Advanced AI Texas Hold'em Poker Bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "websockets",
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "tqdm",
    ],
)
