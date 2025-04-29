from setuptools import setup, find_packages

setup(
    name='crowdsurfer_v2',
    version='2.0.0',
    packages=find_packages(include=['models*']),
    package_dir={'': '.'},
    # packages=['dataloaders',
    #           'models',
    #           'train',
    #           'viz',
    #           'utils',
    #           'runs'],
)