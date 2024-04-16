from setuptools import setup

setup(
    name='IsoMarker',
    version='0.2.0',    
    description='IsoMarker',
    url='https://github.com/davidchen0420',
    author='David Chen',
    author_email='davidc.chen@mail.utoronto.ca',
    license='BSD 2-clause',
    packages=['isomarker'],
    install_requires=['pandas>=1.5.3',
                      'numpy>=1.24.2',   
                      'anndata>=0.10.2',
                        'scanpy>=1.10.1',
                        'matplotlib>=3.8.0',
                        'seaborn>=0.13.0',
                        'scikit-learn>=1.3.1',
                        'pydeseq2>=0.4.8',
                        'shap>=0.44.0',
                        'xgboost>=2.0.3',
                        'jenkspy>=0.4.0',
                        'scipy>=1.13.0',
                        'UpSetPlot>=0.9.0',
                        'adjustText>=1.1.1',
                        'jax==0.4.26',
                        'jaxlib==0.4.26'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)