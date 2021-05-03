from sys import stdout
from datetime import datetime


class ProgressBar:
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.progress_start = datetime.now()

    def start(self):
        self.progress_start = datetime.now()
        self._show()

    def next(self):
        self.current_iteration += 1
        self._show()

    def end(self):
        self._show()
        print("")  # add a newline after the progressbar finished

    def _show(self):
        barlength = 30
        progress = self.current_iteration / self.total_iterations
        progress_count = int(progress * barlength)

        out = "\r[{}{}]".format('#' * progress_count, '.' * (barlength - progress_count))
        out += "[ {}{}/{} eta {}]".format(' ' * (len(str(self.total_iterations)) - len(str(self.current_iteration))),
                                          self.current_iteration, self.total_iterations,
                                          self._convert_to_time(self._calculate_eta()))
        # write some spaces at the end to overwrite old content
        stdout.write(out + 10 * ' ' + 10 * '\b')
        stdout.flush()

    def _calculate_eta(self):
        execution_time = (datetime.now() - self.progress_start).total_seconds()
        if self.current_iteration:
            eta = (self.total_iterations / self.current_iteration - 1) * execution_time
        else:
            eta = 1e5
        return eta

    def _convert_to_time(self, time_duration):
        duration = ""
        hours = int(time_duration / 3600)
        minutes = int(time_duration / 60 - hours * 60)
        seconds = time_duration - hours * 3600 - minutes * 60

        if hours:
            duration += "{}h ".format(hours)
        if minutes:
            duration += "{}m ".format(minutes)
        if seconds:
            duration += "{:.3f}s ".format(seconds)
        else:
            duration += "done "
        return duration
