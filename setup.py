from setuptools import setup, find_packages
from aeromoc import __name__, __version__

# with open('README.md') as f:
#       long_description = f.read()

setup(name=__name__,
      version=__version__,
      description='method of characteristic for flowfield',
      keywords=['MOC', 'nozzle design'],
      # download_url='https://github.com/swayli94/cfdpost/',
      license='MIT',
      author='Aerolab',
      author_email='yyj980401@126.com',
      packages=find_packages(),
      install_requires=['numpy'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ]
)

