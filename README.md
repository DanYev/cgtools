reforge
==============================

## MD pipeline for SOL and PHX clusters

### Installation

1. **Load modules:**

    ```bash
    module load mamba
    module load gromacs
    
    ```

2. **Clone this repository and install the environment:**

   ```bash 
   git clone https://github.com/DanYev/reforge.git
   mamba env create -n reforge --file environment.yml
   source activate reforge
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


