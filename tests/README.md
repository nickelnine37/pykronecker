# Instructions for running tests with PyKronecker

In order to run tests for PyKronecker, the requirements listed in the `requirements_dev.txt` file must be fulfilled. Since Jax is required to run the full test suite, this means testing is only available on Linux and MacOS at the moment. 

## Testing with PyTest

Once the required packages have been installed, the tests can be performed by running the command 

```
pytest
```

in the root folder. This will generate a coverage report which can be found in the `htmlcov` directory. To view it run

```
cd htmlcov && python -m http.server
```

and open the localhost link in a browser.

## Testing with Tox

To test PyKronecker in a clean environment for all python versions from 3.6-3.10, we use Tox. This can be achieved by running 

```
tox
```

in the root directory. Note that this takes significantly longer to run, so is best performed as a final check. 

## Testing with GitHub actions

Whenever code is pushed to the remote repository, the Tox test suite is automatically run using GitHub actions. To investigate this process, consult the file found at `.github/workflows/tests.yml`. 