Home
====

Welcome to the home page of the reForge Python package.

Overview of reForge Package
---------------------------


   The reForge source code is hosted on GitLab at: https://github.com/DanYev/reforge

The reForge Python package provides a comprehensive suite of utilities designed for high‐performance 
simulation, analysis, and data processing workflows. It features optimized mathematical routines, 
GPU‐accelerated computations, and a variety of helper tools to streamline file I/O and data manipulation. 
Developed to support advanced research projects in simulation and analysis, reForge is available for 
public use under the GNU V3 License.

What's in reForge?
------------------

In a nutshell, the key components of reForge are:

- **Optimized Mathematical Routines:**  
  High-performance functions written in Cython and CUDA for computing Hessian matrices, perturbation 
  matrices, and other advanced mathematical operations.

- **GPU-Based Simulation Utilities:**  
  Tools that leverage GPU acceleration for simulation and analysis tasks, significantly reducing 
  computation times.

- **File I/O and Data Processing Helpers:**  
  Functions for reading and writing various data formats (e.g., CSV, NPY, XVG) and for managing 
  file systems efficiently.

- **Profiling Tools:**  
  Decorators and utilities for memory and time profiling to ensure optimal performance during 
  computations.

- **Support for Molecular Dynamics and Diffraction Analysis:**  
  Specialized algorithms and workflows tailored to the simulation of x-ray diffraction under the Born 
  approximation and related applications.

For New Users
-------------

- **Learn the Basics:**  
  Familiarize yourself with shell scripting, git, and Python. Essential packages include 
  `NumPy <https://numpy.org/>`_, `Cython <https://cython.org/>`_, and `CuPy <https://cupy.dev/>`_.

- **Understand the Fundamentals:**  
  Get a grasp of object-oriented programming and high-performance computing concepts to make the 
  most of reForge’s capabilities.

- **Explore the Documentation:**  
  Skim through the available documentation and examples provided within the package to learn about 
  the various tools and workflows.

- **unit conventions:**  
  reforge uses si units throughout (e.g., angles in radians) to maintain consistency in calculations.

for developers
--------------

- **contribution guidelines:**  
  if you plan to contribute to reforge, please refer to the developer documentation for coding 
  standards, testing procedures, and version control guidelines.

- **testing and documentation:**  
  an extensive suite of tests and detailed documentation accompanies the package to ensure 
  reliability and maintainability.

- **community and support:**  
  contributions are welcome! please check the gitlab repository for issues, feature requests, and 
  further discussion.

Acknowledgements
----------------

The reForge package is maintained by **[Your Name or Your Organization]**. 
This project is inspired by and builds upon multiple excellent open-source 
packages such as Cython, NumPy, CuPy, GROMACS, OpenMM, Vermouth and MDAnalysis. 

Indices and Tables
------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. toctree::
   :maxdepth: 2
   :glob:

   getting_started
   why
   reforge
