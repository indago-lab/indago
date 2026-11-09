# Packaging instructions

Do the following **in given order**:

1. Change the Indago version in `pyproject.toml`.
2. Change the Indago version in `src/indago/__init__.py`.
3. Change the Indago version in `docs/src/conf.py`. **NOT NEEDED - WORKS AUTOMATICALLY** (works automatically by importing indago and using `indago.__version__`)
4. Change the Indago version in `docs/src/index.rst`.
5. Commit changes to `master` branch. 
6. Merge `master` to `release` branch in order to release a new verion on pypi.
7. Login to readthedocs.org; Select project Indao, and click "Rebuild version" (under "release"). 


