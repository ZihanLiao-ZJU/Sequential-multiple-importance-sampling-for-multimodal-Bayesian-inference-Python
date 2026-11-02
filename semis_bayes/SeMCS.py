"""Legacy entry point backed by the one semcs_core.Engine."""
from semcs_core import LegacyFacade
from .method import BayesMethod

class SeMCS(LegacyFacade):
    default_verbose=True
    def __init__(self, Sampler):
        super().__init__(Sampler, BayesMethod())
