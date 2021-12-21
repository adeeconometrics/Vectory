.. Vectory documentation master file, created by
   sphinx-quickstart on Wed Dec 22 00:57:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vectory's documentation!
===================================

`Vectory` is a dependency-free vectory algebra library in Python. 
`Vectory` supports elementary operations involoving vectors.
Motivated to solve the problems of my girlfriend, some functions are specialized for ℝ³`.
More importantly, the core data structure is stored in `vector` class which can be wrapped as 
a numpy array  via `np.array(vector)` thereby extending its reach for more complicated expressions.
The intent of `Vectory` is to serve the basic needs for solving small instances of vector algebra problems
without incurring unecessary penalties of bloat dependencies. 


.. image:: ../img/logo.png
   :width: 500

.. toctree::
   :maxdepth: 2
   :caption: Contents:
      vector