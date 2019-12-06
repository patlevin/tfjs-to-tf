import setuptools
from tfjs_graph_converter.version import VERSION 

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tfjs_graph_converter",
    version=VERSION,
    author="Patrick Levin",
    author_email="vertical-pink@protonmail.com",
    description="A tensorflowjs Graph Model Converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patlevin/tfjs-to-tf/",
    install_requires=['tensorflowjs>=1.3.2'],
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            "tfjs_graph_converter = tfjs_graph_converter.converter:pip_main",
        ]
    },
    keywords="tensorflow tensorflowjs converter",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    py_modules=[
        "tfjs_graph_converter",
        "tfjs_graph_converter.version",
        "tfjs_graph_converter.api",
        "tfjs_graph_converter.common",
        "tfjs_graph_converter.converter",
        "tfjs_graph_converter.util",
    ],
    python_requires=">=3.6"
)