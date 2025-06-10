import json
import os
import subprocess
from pathlib import Path

from anytree import Node, RenderTree
from anytree.exporter import DotExporter

from src.tools.path_utils import get_default_results_directory


def print_experiment_log(experiment_name, strong_model, weak_model, problem_number):
    results_dir = get_default_results_directory()
    experiment_dir = results_dir / experiment_name

    # Get all step directories
    step_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

    # Sort step directories by step number and exclude "merged_results"
    step_dirs = [d for d in step_dirs if d.name != "merged_results"]
    step_dirs.sort(key=lambda x: int(x.name.split("-")[0]))

    for step_dir in step_dirs:
        # Check for both strong and weak model directories
        for model in [strong_model, weak_model, f"{strong_model}+{weak_model}"]:
            model_dir = step_dir / model
            if not model_dir.exists():
                continue

            ignore_keys = ["metadata", "prompt", "response"]
            if model == f"{strong_model}+{weak_model}":
                ignore_keys.extend(["question", "test_cases", "uid"])

            # Find matching problem files
            problem_files = list(model_dir.glob(f"{problem_number}*.json"))

            for problem_file in problem_files:
                with open(problem_file, "r") as f:
                    data = json.load(f)

                print(f"Step: {step_dir.name}")
                print(f"Model: {model}")
                print(f"Problem: {problem_file.stem}")
                if not isinstance(data, list):
                    data = [data]

                print("\nPrompt:")
                prompt_array = data[0].get("prompt")
                if prompt_array is None:
                    print("No prompt available")
                else:
                    for text in prompt_array:
                        print(f"Role: {text['role']}")
                        print(f"Content: {text['content']}")

                for response in data:
                    print("\nResponse:")
                    print(
                        response.get("response", {}).get(
                            "completion", "No response available"
                        )
                    )
                    print("\nOther Fields:")
                    for key, value in response.items():
                        if key not in ignore_keys:
                            print(f"{key}: {value}")
                    print("\n" + "=" * 50 + "\n")


def show_pipeline_graph(pipeline):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for task in pipeline.steps:
        G.add_node(task.name)
        for dep in task.dependencies:
            G.add_edge(dep.name, task.name)

    # Print the graph structure
    print("Pipeline Dependency Graph:")
    for node in nx.topological_sort(G):
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        print(f"{node}:")
        if predecessors:
            print(f"  Parents: {', '.join(predecessors)}")
        if successors:
            print(f"  Children: {', '.join(successors)}")

    # Generate a DOT file for visualization
    output_dir = get_default_results_directory() / pipeline.config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_file = output_dir / "pipeline_graph.dot"
    png_file = output_dir / "pipeline_graph.png"

    nx.drawing.nx_pydot.write_dot(G, str(dot_file))
    print(f"DOT file generated at: {dot_file}")

    # Generate PNG file using Graphviz
    try:
        subprocess.run(["dot", "-Tpng", str(dot_file), "-o", str(png_file)], check=True)
        print(f"PNG file generated at: {png_file}")
    except subprocess.CalledProcessError:
        print(
            "Error: Failed to generate PNG. Make sure Graphviz is installed and accessible in your PATH."
        )
    except FileNotFoundError:
        print(
            "Error: Graphviz not found. Please install Graphviz to generate PNG files."
        )

    # Optionally, you can also use matplotlib to visualize the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=8,
        arrows=True,
    )
    plt.title("Pipeline Dependency Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_dir / "pipeline_graph_matplotlib.png"))
    print(
        f"Matplotlib graph generated at: {output_dir / 'pipeline_graph_matplotlib.png'}"
    )
