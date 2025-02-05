from setuptools import setup, find_packages

setup(
    name='sentiment-analysis-api',
    version='1.0.0',
    description='Sentiment Analysis Machine Learning API',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=1.2.2',
        'numpy>=1.24.3',
        'flask>=2.3.2',
        'joblib>=1.2.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'coverage',
            'flake8',
        ],
        'prod': [
            'gunicorn',
            'prometheus-client',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'train-model=scripts.train_model:main',
        ],
    },
)