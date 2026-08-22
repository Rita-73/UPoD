"""UPoD: Uniform-target Population Debiasing for generative recommendation."""

__version__ = "0.1.0"

from genrec import models
from genrec import modules
from genrec import data
from genrec import trainers

__all__ = ["models", "modules", "data", "trainers", "__version__"]
