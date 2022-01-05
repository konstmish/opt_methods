import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='opt_methods', 
    version='0.1.1',
    author='Konstantin Mishchenko',
    author_email='konsta.mish@gmail.com',
    description='A collection of optimization methods and'
                'loss functions for comparing their'
                'iteration convergence and plotting the results',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(),
    url='https://github.com/konstmish/opt_methods'
)
