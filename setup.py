from setuptools import setup, find_packages

setup(
    name="EasyOptimization",
    version="0.1.0",
    author="Sébastien M. Popoff",
    author_email="sebastien.popoff@espci.fr",
    description="A simple optimization algorithm toolbox for wavefront shaping",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wavefrontshaping/EasyOptimization",
    packages=find_packages(),
    install_requires=["numpy", "tqdm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
)
