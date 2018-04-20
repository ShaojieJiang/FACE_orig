from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os

def _path(opt, subtask=None):
    # Build the data if it doesn't exist.
    build(opt, subtask)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'Reddit', subtask, '{type}_{subtask}.txt'.format(type=dt, subtask=subtask))


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)

class AtheismQATeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'atheism_QA')
        super().__init__(opt, shared)

class PoliticsQATeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'politics_QA')
        super().__init__(opt, shared)

class ProgrammingQATeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'programming_QA')
        super().__init__(opt, shared)

class OpenTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'open')
        super().__init__(opt, shared)

class AtheismTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'atheism')
        super().__init__(opt, shared)

class ProgrammingTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'programming')
        super().__init__(opt, shared)

class PoliticsTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'politics')
        super().__init__(opt, shared)
