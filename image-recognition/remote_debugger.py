import pydevd_pycharm

pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
