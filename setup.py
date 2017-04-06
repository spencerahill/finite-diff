import setuptools

setuptools.setup(
    name="indiff",
    version="0.0",
    packages=setuptools.find_packages(),
    author="Spencer A. Hill",
    author_email="shill@atmos.ucla.edu",
    description="Xarray-based finite differencing of gridded geophysical data",
    install_requires=['numpy >= 1.7',
                      'toolz >= 0.7.2',
                      'cloudpickle >= 0.2.1',
                      'xarray >= 0.9.1'],
    tests_require=['pytest >= 2.7.1'],
    license="Apache",
    keywords="climate science, xarray, finite differencing",
    url="https://github.com/spencerahill/infinite-diff",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ]
)
