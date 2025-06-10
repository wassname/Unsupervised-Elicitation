How to use the pipeline:
First, outline the graph you would like to execute, including the following:
* Data Loading
* Model queries
* Code Execution Eval
* Transformations
* Monitoring

Next, convert each of the nodes in that graph into the corresponding helper function:
* add_load_data_step
* add_query_step
* add_code_evaluation_step
* add_transformation_step
* add_monitoring_step

Each of these takes different parameters that you can see in the method signatures. The important ones to know are these:
LoadData takes either a data-loading function and a location, or it takes a dataset
Queries take a prompt function that they pass the incoming data into to create the associated prompt, and a parse function that they use to parse the LLM response
Code Evals take an executor function that executes all of the code in the Solution objects on the associated test cases.
Transforms take arbitrary functions that they apply to the data as a whole.
Monitoring steps take arbitrary monitoring steps. I may eventually enforce that all pipelines end in one of these because it's really what we care about.

Finally, put the dependencies of each step into their dependencies parameter. This is how execution order is determined and how data flows between steps. If you rely on more than one step, the data will be passed to ordered args in the same order as the list of dependencies.

You will also need to include a PipelineConfig parameter that contains metadata around how many concurrents to use and similar.

Once you have a pipeline definition, call the pipeline.run() function on it to execute the graph. This method returns the Pipeline.Results object back, which holds the output of each step in a dictionary.
