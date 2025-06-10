__all__ = [
    "EvalConfig",
    "EvalConfigBuilder",
    "evaluate_solutions",
    "examine_solution",
    "print_eval",
]

import json
import os

from src.code_evaluation.test_results import Solution

import src.tools.path_utils as path_utils

DEFAULT_RESULTS_DIR = path_utils.get_default_results_directory()


class EvalConfig:
    def __init__(
        self,
        experiment_name,
        model_to_test,
        executor_fn,
        language,
        use_cache=True,
        data=None,
        dataloader_fn=None,
        data_location=None,
    ):
        self.experiment_name = experiment_name
        self.model_to_test = model_to_test
        if dataloader_fn is not None:
            self.data = dataloader_fn(data_location)
        elif isinstance(data, str):
            with open(json.load(data), "r") as f:
                self.data = json.load(f)
        else:
            self.data = data
        self.executor_fn = executor_fn
        self.language = language
        self.use_cache = use_cache

    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


class EvalConfigBuilder:
    def __init__(self):
        self.experiment_name = None
        self.model_to_test = None
        self.executor_fn = None
        self.language = None
        self.data = None
        self.use_cache = None
        self.dataloader_fn = None
        self.data_location = None

    def with_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name
        return self

    def with_executor_fn(self, executor_fn):
        self.executor_fn = executor_fn
        return self

    def with_language(self, language):
        self.language = language
        return self

    def with_model_to_test(self, model_to_test):
        self.model_to_test = model_to_test
        return self

    def with_use_cache(self, use_cache):
        self.use_cache = use_cache
        return self

    def with_data(self, data):
        self.data = data
        return self

    def with_dataloader_fn(self, dataloader_fn):
        self.dataloader_fn = dataloader_fn
        return self

    def with_data_location(self, data_location):
        self.data_location = data_location
        return self

    def build(self):
        assert self.experiment_name is not None, "Experiment name must be set"
        assert self.executor_fn is not None, "Executor function must be set"
        assert self.language is not None, "Language must be set"
        assert (self.data is not None) or (
            self.dataloader_fn is not None and self.data_location is not None
        ), "Data must be set"
        return EvalConfig(
            self.experiment_name,
            self.model_to_test,
            self.executor_fn,
            self.language,
            self.use_cache,
            self.data,
            self.dataloader_fn,
            self.data_location,
        )


def evaluate_solutions(eval_config):
    if isinstance(eval_config.data, list):
        eval_config.data = {
            index: response[0] for index, response in enumerate(eval_config.data)
        }

    eval_data = []
    for problem_id, response in eval_config.data.items():
        if response == {}:
            item = Solution.no_solution(problem_id)
        else:
            item = Solution.from_response(problem_id, response, eval_config.language)
        eval_data.append(item)

    results_dir = DEFAULT_RESULTS_DIR
    save_dir = results_dir / eval_config.experiment_name / eval_config.model_to_test
    os.makedirs(save_dir, exist_ok=True)

    # load caching results
    if eval_config.use_cache and save_dir is not None:
        for idx, solution in enumerate(eval_data):
            if solution.correct is not None:  # already have runtime eval results
                continue
            save_path = f"{save_dir}/{solution.question_id}.json"
            if os.path.exists(save_path):
                eval_data[idx] = Solution.from_cache_file(save_path)

    executor_results = eval_config.executor_fn(eval_data)

    for problem_id, problem_data in eval_config.data.items():
        if problem_id not in executor_results:
            executor_results[problem_id] = {}
        for field, value in problem_data.items():
            if field not in executor_results[problem_id]:
                executor_results[problem_id][field] = value

    # save caching results
    if save_dir is not None:
        for question_id, solution in executor_results.items():
            save_path = f"{save_dir}/{question_id}.json"
            with open(save_path, "w") as f:
                json.dump(solution, f, indent=2)

    return executor_results


def examine_solution(solutions, index):
    print(
        f"Difficulty:\n{solutions[index][0]['metadata']['difficulty']}\n----------------------------"
    )
    print(solutions[index][0]["metadata"]["question"])
    print(solutions[index][0]["solution"])
    print("test cases:")
    for test in solutions[index][0]["metadata"]["test_cases"]:
        print(f"{test['input']}{test['output']}")


def print_eval(results):
    correct_tests = 0
    total_tests = 0
    correct_problems = 0
    total_problems = len(results.keys())
    for problem_id, problem_result in results.items():
        correct_tests_local = sum(
            [
                1
                for test_result in problem_result["test_cases"]
                if test_result["correct"]
            ]
        )
        total_tests_local = len(problem_result["test_cases"])
        # print(f"Problem ID: {problem_id}\nTest Results:\n\tCorrect: {correct_tests_local}\n\tTotal: {total_tests_local}\n\tAccuracy: {(correct_tests_local * 100.)/total_tests_local}\nOverall Correct: {problem_result.correct}")
        correct_tests += correct_tests_local
        total_tests += total_tests_local
        if problem_result["correct"]:
            correct_problems += 1
    print(
        f"Number of Problems: {total_problems}\nNumber Correct: {correct_problems}\nAccuracy: {(correct_problems * 100.)/total_problems}\nNumber of Tests: {total_tests}\nNumber Correct: {correct_tests}\nAccuracy: {(correct_tests * 100.)/total_tests}"
    )
