Check a projection derivative
=============================

For an affine equality the projection Jacobian is known analytically. This
example checks the JAX derivative against that matrix, independently of training.

.. literalinclude:: derivatives.py
   :language: python
   :linenos:

Download: :download:`derivatives.py <derivatives.py>`.
