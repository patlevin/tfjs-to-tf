Module ``tfjs_graph_converter.converter``
=========================================

The module ``tfjs_graph_converter.converter`` contains the CLI script and
the main converter function.

Although the module is intended for use as a command-line utility, you can
invoke it from Python, too. The arguments of the function are identical to
the CLI.

.. contents:: **Table of Contents**
    :backlinks: none

``convert``
^^^^^^^^^^^

.. code:: Python

    convert(
        arguments: List[str]
    ) -> None

This function accepts command-line arguments given as a list of strings
and performs the requested actions. Some command line switches and options
aren't very useful when called from within another script, though
(e.g. ``--help``).

..

    **Arguments**

**arguments**
    List of command-line argument strings:

    ========================== ==============================================
    Option/Switch              Description
    ========================== ==============================================
    --output_format            Selects the output format that the graph modul
                               should be comverted to. There are two possible
                               choices:

                               **tf_frozen_model** ``[default]``
                                    Convert the TFJS model to a TF frozen
                                    graph model, e.g. a single ``.pb`` file.
                               **tf_saved_model**
                                    Convert the TFJS model to a TF1
                                    ``SavedModel``. The output path must be a
                                    directory if this option is selected
    -------------------------- ----------------------------------------------
    --saved_model_tags         Defines the tags attached to the
                               ``SavedModel`` if the selected output format
                               is ``tf_saved_model``.
                               Multiple tags can be given as a
                               comma-separated list.
    -------------------------- ----------------------------------------------
    --compat_mode, -c          Keep the input types compatible with
                               TensorflowJS <=v2.4.x
    -------------------------- ----------------------------------------------
    --version, -v              Prints the library version and the versions of
                               dependencies (TF, TFJS). *Useful only in CLI*
    -------------------------- ----------------------------------------------
    --silent, -s               Suppress any text output besides errors and
                               warnings. *Useful only in CLI*
    ========================== ==============================================

    Values for options can be set by either passing them as a single string
    with an equal sign (``=``) separating option and value (e.g.
    ``--output_format=tf_frozen_model``). Or you can pass them as two
    consequitive items in the arguments-list (e.g.
    ``['--output_format', 'tf_saved_model']``).

    Any list item not preceded by an option that requires a second list item
    as its value, is considered a positional argument. The two required
    positional arguments are:

    1. **Input model directory** - holds the path that contains the TFJS
       ``model.json`` and weight files. If the model description is named
       differenttly, the file name can be provided directly.
       All model weights must be in the same directory as the model JSON,
       though.

    2. **Output model path** - denotes the output file name and -path.
       For ``tf_frozen_model`` this is the path and name of a ``.pb``-file.
       In case the selected output format is ``tf_saved_model``, this is
       the name of an (empty) directory in which the ``SavedModel`` and its
       artifacts (meta graph, weights, and index) will be written.

..

    **Example**

.. code:: python

    import tfjs_graph_converter.converter as tfjs

    # convert to SavedModel
    tfjs.convert([
        '--output_format=tfjs_graph_model_folder'
        '--saved_model_tags', 'serving_default,cpu'
        './models/tfjs_graph_model_folder/',
        './prod/models/saved_model/1/'
    ])
