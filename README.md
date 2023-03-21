# mlflow-examples

| Example                              | Description                                                                        |
|--------------------------------------|------------------------------------------------------------------------------------|
| background_job_example               | MLFlow Background Job On Anaconda Enterprise                                       |
| background_workflow_example          | Synchronous MLFlow Multi-Step Workflow With Background Jobs On Anaconda Enterprise |
| background_workflow_parallel_example | MLFlow Multi-Step Workflow With Parallel Background Jobs On Anaconda Enterprise    |

### How To Deploy An Example

```commandline
anaconda-project run ae5 project upload --name {EXAMPLE NAME} {LOCAL PATH TO EXAMPLE}
```

For a specific example the command would look like these:
```commandline
anaconda-project run ae5 project upload --name background_job_example examples/background_job_example
```

```commandline
anaconda-project run ae5 project upload --name background_workflow_example examples/background_workflow_example
```
