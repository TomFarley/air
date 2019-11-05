======================================
Using FIRE with data on other tokamaks
======================================

Reading IR movie data
---------------------

TODO: Complete this documentation

In order to read movie data from a format that is not natively supported, update your `~/.fire.conf` file to include
the path to a python module containing the functions `read_movie_meta` and `read_movie_data`.

.. code-block:: python
    :caption: ~/.fire.conf
    :name: .fire.conf
    :emphasize-lines: 6

    ...
    {
        "mast": "{fire_dir}/interfaces/mast.py",
        "mast_u": "{fire_dir}/interfaces/mast_u.py",
        "jet": "{fire_dir}/interfaces/jet.py",
        "<my_machine>": "<path_to_my_script>/<my_machine>.py",
        ...
    }
    ...

.. autofunction:: fire.interfaces.movie_format_template.read_movie_meta

This function should return a dictionary with the following elments:

.. code-block:: python

    movie_meta = {}


.. autofunction:: fire.interfaces.movie_format_template.read_movie_data