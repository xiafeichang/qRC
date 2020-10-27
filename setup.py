from distutils.core import setup

# Check if ROOT is installed and raise error if not found
try:
    import ROOT
except ModuleNotFoundError:
    raise ImportError('ROOT not found. You need a full working installation of ROOT to install this package.\n' \
            'For more info, see: https://root.cern/install/')

setup(
    name = "quantile_regression_chain",
    author = "Thomas Reitenspiess",
    packages = ['quantile_regression_chain'],
)
