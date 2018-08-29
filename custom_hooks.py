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
from utils import save_images, append_index

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
        fetches={
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

