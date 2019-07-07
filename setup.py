import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TLSynDataGene",
    version="0.0.1",
    author="Sergio Peignier, Mounir Atiq",
    author_email="sergio.peignier@insa-lyon.fr, atiq.mounir@gmail.com",
    description="Synthetic data generator to test transfer learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)

print(setuptools.find_packages())
