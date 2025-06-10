__all__ = ["load_prompts", "load_problems", "load_problems_from_json", "load_solutions"]

import json
import logging
import os

from .path_utils import get_default_results_directory, get_root_directory

logger = logging.getLogger(__name__)

ROOT_DIR = get_root_directory()
DATA_DIR = ROOT_DIR / "data" / "APPS"
PROMPTS_DIR = ROOT_DIR / "src" / "prompts"


def get_data_dir():
    return DATA_DIR


def load_prompts(prompt_type):
    files = [f for f in (PROMPTS_DIR / prompt_type.value).glob("*") if f.is_file()]
    prompts = {}
    for file in files:
        with file.open("r") as f:
            prompts[file.name] = f.read()
    return prompts


def load_problem_subset(subset, require_solutions=False, problem_ids=None):
    def load_problems(dir, num_problems=None):
        if problem_ids:
            problem_dirs = problem_ids
        else:
            problem_dirs = os.listdir(dir)
        problems = {}
        added = 0
        for problem_dir in problem_dirs:
            problem_path = dir / problem_dir
            problem = {}
            with (problem_path / "metadata.json").open("r") as f:
                problem["metadata"] = json.load(f)
            if (
                subset != "ALL"
                and problem["metadata"]["difficulty"].lower() != subset.lower()
            ):
                continue

            if require_solutions and not (problem_path / "solutions.json").exists():
                logger.debug(
                    f"Skipping problem {problem_dir} because it does not have solutions"
                )
                continue

            with (problem_path / "question.txt").open("r") as f:
                problem["question"] = f.read()

            problem["uid"] = problem_dir
            problems[problem_dir] = problem
            added += 1
            if added >= num_problems:
                break

        return problems

    return load_problems


def load_problems(dir, num_problems=None):
    return load_problem_subset("ALL")(dir, num_problems)


def load_problems_from_json(path, num_problems=None, problem_ids=None):
    problems = {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print('read data error: ', e)
        data = path
    if num_problems is not None:
        data = data[:num_problems]

    for i, item in enumerate(data):
        item["uid"] = i
        if 'vanilla_label' not in item:
            item["vanilla_label"] = item["label"]
        problems[f"{i}"] = item 
    return problems


def load_problems_from_json_ids(path, num_problems=None, problem_ids=None):
    problems = {}
    try:
        with open(path) as f:
            data = json.load(f)
    except:
        data = path
        
    if problem_ids is not None:
        data = [data[i] for i in problem_ids]

    for i, item in enumerate(data):
        item["uid"] = i
        if 'vanilla_label' not in item:
            item["vanilla_label"] = item["label"]
        # acc.append(item['label'] == item['vanilla_label'])
        problems[f"{i}"] = item

    return problems


def load_assignments(path, num_problems=None, problem_ids=None):
    return path


def load_solutions(dir, num_problems=None):
    solutions = {}
    files = dir.glob("*")
    files = [f for f in files if f.name != "incoming_problem_ids.json"]
    if num_problems is not None:
        files = list(files)[:num_problems]

    for file in files:
        with file.open("r") as f:
            solution = json.load(f)
            # Assume 1 solution per file
            if isinstance(solution, list):
                solutions[file.stem] = solution[0]
            else:
                solutions[file.stem] = solution
    return solutions


def load_multiple_solutions(dir, num_problems=None, problem_ids=None):
    solutions = {}
    files = dir.glob("*")
    files = [f for f in files if f.name != "incoming_problem_ids.json"]

    for file in files:
        with file.open("r") as f:
            solution = json.load(f)
            if "metadata" in solution:
                solution.pop("metadata")
            if "demonstration" in solution:
                solution.pop("demonstration")
            solutions[file.stem] = solution
    return solutions

def load_multiple_solutions_w2s(dir, num_problems=None, problem_ids=None):
    solutions = {}
    files = dir.glob("*")
    files = [f for f in files if f.name != "incoming_problem_ids.json"]

    for file in files:
        with file.open("r") as f:
            solution = json.load(f)
            metadata = solution[0]['metadata']
            if "demonstration" in metadata:
                metadata.pop("demonstration")
            metadata['label'] = solution[0]['score'] > 0
            solutions[file.stem] = metadata
    return solutions


def save_to_cache(data, name, delete_existing=False, incoming_problem_ids=None):
    dir = get_default_results_directory() / name

    # Delete all files in the directory first
    if delete_existing and os.path.exists(dir):
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    os.makedirs(dir, exist_ok=True)
    for k, v in data.items():
        if isinstance(v, list):
            to_write = [
                {
                    key: value
                    for key, value in item.items()
                    if key not in ["prompt", "response"]
                }
                for item in v
            ]
        else:
            to_write = {
                key: value
                for key, value in v.items()
                if key not in ["prompt", "response"]
            }
        with open(dir / f"{k}.json", "w") as f:
            json.dump(to_write, f, indent=4)

    if incoming_problem_ids:
        with open(dir / "incoming_problem_ids.json", "w") as f:
            json.dump({"problem_ids": list(incoming_problem_ids)}, f, indent=4)


def read_from_cache(name):
    dir = get_default_results_directory() / name
    data = {}
    incoming_problem_ids = []

    for file in dir.glob("*.json"):
        if file.name == "incoming_problem_ids.json":
            with file.open("r") as f:
                incoming_problem_ids = json.load(f).get("problem_ids", [])
        else:
            with file.open("r") as f:
                value = json.load(f)
                if not value.get("metadata"):
                    value["metadata"] = {k: v for k, v in value.items()}
                data[file.stem] = value

    return data, incoming_problem_ids


def load_ground_truth_solutions(problem_ids):
    output = {}
    for problem_id in problem_ids:
        with open(get_data_dir() / "test" / problem_id / "solutions.json", "r") as f:
            solutions = json.load(f)
            cleaned_solutions = []
            for solution in solutions:
                # Remove unwanted lines from the solution
                cleaned_solution = []
                for line in solution.split("\n"):
                    if (
                        not line.strip().startswith("#!")
                        and " input=" not in line
                        and "sys.stdin" not in line
                    ):
                        cleaned_solution.append(line)
                cleaned_solutions.append("\n".join(cleaned_solution).strip())
            output[problem_id] = cleaned_solutions
    return output


def load_test_case(problem_id):
    problem_id = problem_id.split("-")[0]
    with open(get_data_dir() / "test" / problem_id / "input_output.json", "r") as f:
        data = json.load(f)
        return [
            {"input": i, "output": o} for (i, o) in zip(data["inputs"], data["outputs"])
        ]


def load_test_cases(problem_ids):
    output = {}
    for problem_id in problem_ids:
        output[problem_id] = load_test_case(problem_id)
    return output


loaded_test_cases = {}


def get_test_cases_for_single_problem(problem_id):
    global loaded_test_cases
    if problem_id not in loaded_test_cases:
        loaded_test_cases[problem_id] = load_test_case(problem_id)

    return loaded_test_cases[problem_id]
