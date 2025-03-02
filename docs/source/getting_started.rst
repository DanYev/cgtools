Getting started
=================

1. **Load modules:**

.. code-block:: bash

    module load mamba
    module load gromacs

2. **Clone this repository and install the environment:**

.. code-block:: bash

    git clone https://github.com/DanYev/reforge.git
    mamba env create -n reforge --file environment.yml
    source activate reforge

3. **Install the package:**

.. code-block:: bash

    pip install -e .

Running the Tutorial
--------------------

To run the tutorial, execute the following command in the tutorial directory:

.. code-block:: bash

    python submit.py

