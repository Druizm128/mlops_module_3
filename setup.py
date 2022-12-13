import setuptools
from setuptools import find_packages
 
setuptools.setup(
    name="ml",
    version="0.0.0",
    description="Starter code.",
    author="Student",
    packages=find_packages('src'),
    package_dir={'': 'src'}
)
