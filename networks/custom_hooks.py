""" cumstom hooks """
import math
import time
import logging
import numpy as np
import tensorflow as tf

class TraceHook(tf.train.SessionRunHook):
    """ Hook to perform Traces every N steps.
        It checks in before_run whether it has to trace or not and if so, adds the RunOptions.
        In after_run it checks if the next run call needs to be traced and if so, it sets _trace
        to True again. Additionally it stores the metadata when it is available.
    """

    def __init__(self, ckptdir, every_n_step, trace_level=tf.RunOptions.FULL_TRACE): #pylint: disable=E1101
        self._trace = every_n_step == 1
        self.writer = tf.summary.FileWriter(ckptdir)
        self.trace_level = trace_level
        self.every_n_step = every_n_step
        #
        self._global_step_tensor = None

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _TraceHook.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None
        return tf.train.SessionRunArgs(fetches=self._global_step_tensor,
                                       options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results - 1
        if self._trace:
            self._trace = False
            self.writer.add_run_metadata(run_values.run_metadata, "step_%d" % global_step)

        if not (global_step + 1) % self.every_n_step:
            self._trace = True

#
from .utils import save_images, append_index

class DisplayHook(tf.train.SessionRunHook):
    """ Hook to save display images every N steps. """

    def __init__(self, display_fetches, out_dir, every_n_step):
        self.display_fetches = display_fetches
        self.out_dir = out_dir
        self.every_n_step = every_n_step

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _TraceHook.")

    def before_run(self, run_context):
        fetches = {
            "display":self.display_fetches,
            "global_step": tf.train.get_global_step()
        }
        return tf.train.SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        global_step = run_values.results["global_step"]
        images = run_values.results["display"]
        if not global_step % self.every_n_step:
            print("saving display images")
            filesets = save_images(images, self.out_dir, step=global_step)
            append_index(filesets, self.out_dir, step=True)

class NeverTriggerTimer:
    """Timer that never triggers."""

    def should_trigger_for_step(self, step):
        _ = step
        return False

    def update_last_triggered_step(self, step):
        _ = step
        return (None, None)

    def last_triggered_step(self):
        return None

def _as_graph_element(obj):
    """Retrieves Graph element."""

    graph = tf.get_default_graph()
    if not isinstance(obj, str):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj

    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element

# see comments below
tf.logging.set_verbosity(tf.logging.INFO)

class LoggingTensorHook(tf.train.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.
    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:
    ```python
        tf.logging.set_verbosity(tf.logging.INFO)
    ```
    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional inputs.
    """

    def __init__(self, tensors,
                 batch_size, max_steps, steps_per_epoch,
                 every_n_iter=None, every_n_secs=None,
                 at_end=False, formatter=None):
        """Initializes a `LoggingTensorHook`.
        Args:
            tensors: `dict` that maps string-valued tags to tensors/tensor names,
                or `iterable` of tensors/tensor names.
            every_n_iter: `int`, print the values of `tensors` once every N local
                steps taken on the current worker.
            every_n_secs: `int` or `float`, print the values of `tensors` once every N
                seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
                provided.
            at_end: `bool` specifying whether to print the values of `tensors` at the
                end of the run.
            formatter: function, takes dict of `tag`->`Tensor` and returns a string.
                If `None` uses default printing all tensors.

        Raises:
            ValueError: if `every_n_iter` is non-positive.
        """
        only_log_at_end = (at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")

        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)

        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = sorted(tensors.keys())

        self._tensors = tensors
        self._formatter = formatter
        self._timer = (
            NeverTriggerTimer() if only_log_at_end else
            tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))

        self._log_at_end = at_end

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.steps_per_epoch = steps_per_epoch

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self.start = time.time()
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            fetches = {
                "tensor_values":self._current_tensors,
                "global_step": tf.train.get_global_step()
            }
            return tf.train.SessionRunArgs(fetches)
        else:
            return None

    def _log_tensors(self, tensor_values, global_step):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)

        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            tf.logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("%s = %s" % (tag, tensor_values[tag]))

            if elapsed_secs is not None:
                tf.logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
            else:
                tf.logging.info("%s", ", ".join(stats))

        # global_step will have the correct step count if we resume from a checkpoint
        train_epoch = math.ceil(global_step / self.steps_per_epoch)
        train_step = (global_step - 1) % self.steps_per_epoch + 1
        step = self._iter_count
        rate = (step + 1) * self.batch_size / (time.time() - self.start)
        remaining = (self.max_steps - step) * self.batch_size / rate
        print("progress: epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))

        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            tensor_values = run_values.results['tensor_values']
            global_step = run_values.results['global_step']
            self._log_tensors(tensor_values, global_step)

        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            fetches = {
                "tensor_values":self._current_tensors,
                "global_step": tf.train.get_global_step()
            }
            values = session.run(fetches)
            self._log_tensors(values['tensor_values'], values['global_step'])
