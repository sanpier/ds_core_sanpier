from setuptools import setup

setup(
    name='ds-core-sanpier',
    version='0.1.4',
    author='Saner Turhaner',
    author_email='san-pi@windowslive.com',
    packages=['ds_core_sanpi'],
    url='http://pypi.python.org/pypi/ds-core-sanpier/',
    license='MIT',
    description='A package to automize some of the steps before modeling and in the modeling stage',
    long_description=open('README.md').read(),
    install_requires=[
        "catboost >= 1.0.5",
        "imblearn >= 0.0",
        "lightgbm >= 3.3.2",
        "lofo-importance >= 0.3.1",
        "matplotlib >= 3.3.4",
        "numpy >= 1.19.5",
        "pandas >= 1.1.5",
        "seaborn >= 0.11.1",
        "scikit-learn >= 0.24.2",
        "shap >= 0.40.0"
    ]
)