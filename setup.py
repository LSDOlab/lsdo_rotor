from setuptools import setup, find_packages
import os.path
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lsdo_rotor',
    version=get_version('lsdo_rotor/__init__.py'),
    author='Marius Ruh',
    author_email='mruh@ucsd.edu',
    license='LGPLv3+',
    keywords='rotor analysis, blade element momentum',
    url='https://github.com/LSDOlab/lsdo_rotor',
    download_url='http://pypi.python.org/pypi/lsdo_rotor',
    description='low-fidelity rotor analysis and design with blade element momentum theory',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'pytest',
        'pandas',
        'matplotlib',
        'scipy',
        'csdl @ git+https://github.com/LSDOlab/csdl.git',
        'python_csdl_backend @ git+https://github.com/LSDOlab/python_csdl_backend.git',
        'lsdo_airfoil @ git+https://github.com/LSDOlab/lsdo_airfoil.git',
        'modopt @ git+https://github.com/LSDOlab/modopt.git',
        'm3l @ git+https://github.com/LSDOlab/m3l.git',
        'smt',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
)

