from setuptools import setup, find_packages

setup(name='abides',
      version='1.0.0',
      description='Agent-Based Interactive Discrete Event Simulation',
      url='https://github.com/abides-sim/abides',
      author='davebyrd',
      author_email='dave@imtc.gatech.edu',
      license='BSD 3-Clause License',
      packages=find_packages(),
      install_requires=[
          'cycler',
          'joblib',
          'jsons',
          'kiwisolver',
          'matplotlib',
          'numpy',
          'pandas',
          'pprofile',
          'pyparsing',
          'python-dateutil',
          'pytz',
          'scipy',
          'seaborn',
          'six',
          'tqdm',
          'psutil'
      ]
      )
