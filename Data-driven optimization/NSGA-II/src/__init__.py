"""NSGA-II from-scratch implementation package."""

from .nsga2 import NSGA2, NSGA2Result
from .problems import Problem, get_problem

__all__ = ["NSGA2", "NSGA2Result", "Problem", "get_problem"]
