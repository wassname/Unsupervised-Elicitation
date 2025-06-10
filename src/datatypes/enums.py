__all__ = ["Language", "PromptType"]

from enum import Enum


class Language(Enum):
    PYTHON = ("python", "Python")
    CPP = ("cpp", "C++")

    def __init__(self, code, text):
        self.code = code
        self.text = text

    @staticmethod
    def from_code(code):
        if code == "python":
            return Language.PYTHON
        elif code == "cpp":
            return Language.CPP
        else:
            raise Exception(f"Unknown code langauge: {code}")


class PromptType(Enum):
    SOLUTION = "solution_generation"
    BLUE_TEAM = "blue_team"
    RED_TEAM = "red_team"
    EVAL = "eval"


class DifficultyEstimationType(Enum):
    PROBLEM_ONLY = "problem_only"
    PROBLEM_SOLUTION = "problem_solution"
    PROBLEM_SOLUTION_EXECUTION = "problem_solution_execution"
