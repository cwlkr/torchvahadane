from setuptools import setup

setup(
    name='torchvahadane',
    version='0.1.0',
    description='Gpu accelerated vahadane stain normalization',
    url='https://github.com/cwlkr',
    author='Cédric Walker',
    author_email='walker.cedric@outlook.com',
    license='BSD 2-clause',
    packages=['torchvahadane'],
    install_requires=['torch',
                      'numpy',
                      'opencv-python',
                      'tqdm',
                      'scipy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
