from setuptools import setup


setup(
    name='fcompile',
    version='0.1',
    description='Fast Fortran build tool',
    author='Jan Hermann',
    author_email='dev@hermann.in',
    url='https://github.com/azag0/fcompile',
    packages=['fcompile'],
    scripts=['scripts/fcompile', 'scripts/fconfigure'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Framework :: AsyncIO',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Build Tools',
    ],
    license='Mozilla Public License 2.0',
)
