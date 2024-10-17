pyrotini
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/pyrotini/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/pyrotini/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrotini/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrotini/branch/main)


## CGMD scripts for pyRosetta on SOL cluster

### Installation

1. **Load modules:**

    ```bash
    module load mamba
    module load gromacs
    
    ```

2. **Clone this repository and install the environment:**

   ```bash 
   git clone https://github.com/DanYev/pyrotini.git
   mamba env create -n prttest --file env.yml
   source activate prttest
   ```

3. **Install the package:**

    ```bash
    pip install -e .
    ```

**Find tutorial directory and run:**

```bash
sbatch submit2sol.sh
```


### Copyright

Copyright (c) 2024, DY


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
