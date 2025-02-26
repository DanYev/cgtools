# my_package/__init__.py

import os
import importlib

__all__ = []
do_not_import = ["__init__.py", "insane.py", "martinize_nucleotides_old.py"]

package_dir = os.path.dirname(__file__)
for module in os.listdir(package_dir):
    if module.endswith(".py") and module not in do_not_import:
        module_name = module[:-3]
        imported_module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = imported_module
        __all__.append(module_name)
