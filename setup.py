import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='opt_methods', 
    version='0.1.2',
    author='Konstantin Mishchenko',
    author_email='konsta.mish@gmail.com',
    description='A collection of optimization methods, and'
                'loss functions, and examples of comparing their'
                'iteration convergence',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(),
    url='https://github.com/konstmish/opt_methods'
)
