""" Request Failure Error Definition """


class RequestFailureError(Exception):
    """Request Failure Error"""

    status_code: int
    request: dict

    def __init__(self, message: str, status_code: int, request: dict):
        super().__init__(message)

        self.status_code = status_code
        self.request = request
