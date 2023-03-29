import time
from typing import Any, Dict, List, Union

import mlflow
from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import LocalSubmittedRun, SubmittedRun
from mlflow_adsp import AnacondaEnterpriseSubmittedRun

from anaconda.enterprise.server.common.sdk import demand_env_var_as_int

from ..contracts.dto.execute_step_request import ExecuteStepRequest


def wait_on_workers(workers: List[AnacondaEnterpriseSubmittedRun]) -> None:
    wait_time: int = 10  # seconds
    max_wait_counter: int = 100
    counter: int = 0
    wait: bool = True

    while wait:
        if check_workers_complete(workers=workers):
            wait = False
        else:
            time.sleep(wait_time)
            counter += 1
        if counter >= max_wait_counter:
            wait = False

    if counter >= max_wait_counter:
        raise Exception("Not all workers completed within the defined time frame")


def get_status_of_workers(workers: List[AnacondaEnterpriseSubmittedRun]) -> List[RunStatus]:
    statuses: List[RunStatus] = []

    for worker in workers:
        statuses.append(worker.get_status())

    return statuses


def check_workers_complete(workers: List[AnacondaEnterpriseSubmittedRun]) -> bool:
    statuses: List[RunStatus] = get_status_of_workers(workers=workers)

    running_workers: int = 0
    for status in statuses:
        if status == RunStatus.RUNNING:
            running_workers += 1

    if running_workers != 0:
        print(f"{running_workers} of {len(workers)} workers still running")
        return False

    return True


def get_batches(batch_size: int, source_list: List[str]) -> List[List[str]]:
    """
    Given a batch size and a list of strings, will generate a list of lists of strings split by batch size.

    Parameters
    ----------
    batch_size: int
        The maximum size a batch can be.  It is possible to receive a batch that is smaller than this.
        This value must be greater than zero.
    source_list: List[str]
        A list of strings to split into batches.

    Returns
    -------
    batches: List[List[str]]
        The source list split up by batch size into smaller lists and returned together.
    """

    if batch_size < 1:
        raise ValueError(f"Batch size must be greater than zero.  Saw: ({batch_size})")

    batches: List = []

    while len(source_list) > 0:
        if len(source_list) >= batch_size:
            new_batch: List[str] = source_list[:batch_size]
            source_list[:batch_size] = []
        else:
            new_batch: List[str] = source_list
            source_list = []
        batches.append(new_batch)

    return batches


def execute_step(request: ExecuteStepRequest) -> Union[
    SubmittedRun, LocalSubmittedRun, AnacondaEnterpriseSubmittedRun, Any]:
    """
    Execute a MLFlow Workflow Step

    Parameters
    ----------
    request: ExecuteStepRequest

    Returns
    -------
    submitted_job: Union[SubmittedRun, LocalSubmittedRun, AnacondaEnterpriseSubmittedRun, Any]
        An instance of `SubmittedRun` for the requested workflow step run.
    """
    request_dict: Dict = request.dict(by_alias=False)
    print(f"Launching new background job for: {request_dict}")
    run: Union[SubmittedRun, LocalSubmittedRun, AnacondaEnterpriseSubmittedRun, Any] = mlflow.projects.run(
        **request_dict)
    return run


def wait_on_execute_step(request: ExecuteStepRequest) -> AnacondaEnterpriseSubmittedRun:
    run: AnacondaEnterpriseSubmittedRun = execute_step(request=request)
    run.wait()
    return run


def process_work_queue(jobs: List[ExecuteStepRequest]) -> List[AnacondaEnterpriseSubmittedRun]:
    loop_quantum: int = 5
    max_workers: int = demand_env_var_as_int(name="AE_WORKER_MAX")

    todo: List[ExecuteStepRequest] = jobs
    inprogress: List[AnacondaEnterpriseSubmittedRun] = []
    complete: List[AnacondaEnterpriseSubmittedRun] = []

    process_loop: bool = True
    while process_loop:
        # Fill our processing queue
        print("fill processing queue")
        while len(inprogress) < max_workers and len(todo) > 0:
            new_worker: AnacondaEnterpriseSubmittedRun = execute_step(todo.pop())
            inprogress.append(new_worker)

        # review in progress jobs
        print("reviewing in progress jobs")
        new_inprogress: List[AnacondaEnterpriseSubmittedRun] = []
        while len(inprogress) > 0:
            popped_job: AnacondaEnterpriseSubmittedRun = inprogress.pop()
            if popped_job.get_status() != RunStatus.RUNNING:
                complete.append(popped_job)
            else:
                new_inprogress.append(popped_job)
        inprogress = new_inprogress

        # determine if we are complete
        print("determining if we are complete")
        if len(todo) <= 0:
            process_loop = False

        # allow for processing time when the queue is full and there's work to do.
        print("allowing for processing time when the queue is full")
        if len(todo) > 0 and len(inprogress) >= max_workers:
            print("queue is full, pausing before refilling the worker queue ...")
            time.sleep(loop_quantum)
            print("done")

    return complete
