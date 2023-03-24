from typing import Dict, Optional

from anaconda.enterprise.server.contracts import BaseModel


class ExecuteStepRequest(BaseModel):
    """
#     entry_point: str
#         The workflow step to execute.
#     parameters: Dict
#         The dictionary of parameters to pass to the workflow step.
#     run_id: Optional[str] = None
#         If provided it is supplied and used for reporting.
#     backend: str = "local"
#         Default to `local` unless another is provided.
#     synchronous: bool = True
#         Controls whether to return immediately or after run completion.
#     run_name: Optional[str] = None
#         If provided it is supplied and used for reporting.
#     resource_profile: str
#         The resource profile to run the step on (if using the adsp backend)
    """
    uri: str = "."
    entry_point: str = "main"
    parameters: Optional[Dict] = None
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    backend: str = "local"
    backend_config: Optional[Dict] = None
    synchronous: bool = True
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    env_manager: str = "local"
