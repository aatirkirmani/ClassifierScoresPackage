from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='classifierscores',
  version='0.0.2',
  description='Basic Classifiers with train and validation scores',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Aatir Kirmani',
  author_email='kirmaniaatir@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='model classication scores classifier', 
  packages=find_packages(),
  install_requires=[''] 
)