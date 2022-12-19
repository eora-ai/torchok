How to contribute
#################

Before you get started contributing, you should prepare your environment. It's pretty easy to do since TorchOk uses 
`Poetry`_ - a modern Python packaging and dependency management system. Install it via 
`official instructions <https://python-poetry.org/docs/#installation>`_.

Then you need to install dependencies of TorchOk. Install the latest dependencies that developers used to contribute 
the last time:

.. code-block:: bash

    poetry install

The previous command will install TorchOk's dependencies from `poetry.lock` file. If by some reason the `poetry.lock` 
file is outdated, you can remove it and install dependencies from scratch 
(as specified in `pyproject.toml` file):

.. code-block:: bash

    rm poetry.lock
    poetry install

And finally, install the development and documentation dependencies (they are managed separately to poetry at the 
moment):

.. code-block:: bash

    pip install -r requirements/dev.txt
    pip install -r requirements/docs.txt

Working on an already known issue
*********************************

You can find existing issues on `project's GitHub page <https://github.com/eora-ai/torchok/issues>`. When you found 
an issue that you would like to work on, follow the `code_changes`_.

Submitting your own issue
*************************

We welcome new issues not less than new Pull Requests, so if you see a bug or a new feature opportunity, You can create 
your own issue: 
#. Go to `project's GitHub page <https://github.com/eora-ai/torchok/issues>` and click **New issue** button. You will see a bunch of templates for different types of issues
#. Select a template and click `Get started`
#. Type text of the issue by following the template's guideline

.. _code_changes:

Applying code changes
*********************

As you have your environment ready, you can make changes to the code and submit it:

#. Create a new branch from `main` branch calling it `feature-<name_of_feature>`, `fix-<name_of_fix>` or `docs-<name_of_changes>` depending on what sort of changes you are trying to introduce: feature, bug fix or documentation changes, respectively
#. Make code changes
#. Commit changes adding a simple commit message like `MetricsManager: reset metrics optionally on epoch end`
#. Push the changes to the GitHub repository
#. Create a Pull Request briefly describing your work, it should point from your branch to the `main` branch
#. The automatic tests will run to check that your changes don't break something, and the fresh documentation will be built for your branch (you can check it by the link displayed in the corresponding workflow result of the PR)
#. Wait until your PR is reviewed. If something might be done better, you will be asked to perform changes
#. After all changes are done, and PR looks good for the reviewers (at least 1 approval is needed), your PR will be merged into the `main` branch. You can see your changes in the next release of the library

.. note::

    If you are changing documentation, you can rebuild and view it locally:

        .. code-block:: bash

            cd docs
            rm -r build & make html     # remove existing build and build new docs with Sphinx
            python -m http.server       # recommendation: run it in a separate tab for continuous use

.. _Poetry: https://python-poetry.org/
