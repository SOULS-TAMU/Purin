PURIN
=====

Differentiable projections, sparse operators and numerical solvers in JAX.

This guide describes version **1.0.0**. Start with the runnable example,
then use the guide and API reference for your own problem.

Quick overview
--------------

Define the constraint set, choose a solver compatible with its structure, and
prepare a reusable correction layer. Call the layer with candidate outputs and
problem parameters. JAX can differentiate supported projection paths using the
configured derivative policy.

Start with :doc:`affine projection <examples/quickstart>`, compare its
:doc:`analytic Jacobian <examples/derivatives>`, or consult the
:doc:`solver selection guide <SOLVERS>` for affine and nonlinear problems.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   INSTALL
   examples/index
   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   ECOSYSTEM
   MIGRATION
   citations
   license
   contributing
   changelog
   BUILD
   PUBLISHING

Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
