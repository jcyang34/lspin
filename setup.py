from setuptools import setup

setup(name='lspin',
      version='0.1.0',
      description='Locally SParse Interpretable Networks (LSPIN)',
      url='http://github.com/jcyang34/lspin',
      author='Junchen Yang, Ofir Lindenbaum',
      author_email='junchen.k.yang@gmail.com',
      license='MIT',
      packages=['lspin'],
      install_requires=[
          'tensorflow-gpu==1.15.2',
          'optuna',
          'sklearn',
      ],
      zip_safe=False)