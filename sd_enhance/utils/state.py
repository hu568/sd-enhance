"""简化版状态追踪，用于进度显示和中断控制。"""


class State:
    """追踪处理任务的进度，支持中断信号。"""

    def __init__(self):
        self.interrupted = False
        self.job = ""
        self.job_no = 0
        self.job_count = 0
        self.textinfo = None
        self.current_image = None

    def begin(self, job: str = ""):
        self.interrupted = False
        self.job_no = 0
        self.job_count = 0
        self.job = job
        self.textinfo = None
        self.current_image = None

    def end(self):
        self.job = ""
        self.job_count = 0

    def nextjob(self):
        self.job_no += 1

    def interrupt(self):
        self.interrupted = True

    @property
    def is_interrupted(self) -> bool:
        return self.interrupted
