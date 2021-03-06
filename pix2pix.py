""" main module to lauch pix2pix """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import time

import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", default='./datasets/facades/train', help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", type=str, default='./facades_train', help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="BtoA", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
#
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to run")
parser.add_argument("--model", type=str, default='keras', help='network model')

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

a = parser.parse_args()

# set up gpus
# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu_id)

# Note: import * only allowed at module level
# load data
from networks.load_examples import * # pylint: disable=W0614
# load utilities
from networks.utils import * # pylint: disable=wildcard-import,,unused-import,line-too-long

# pick a model
if a.model == 'keras':
    print(' using keras model')
    from networks.model_keras import * # pylint: disable=W0614
else:
    print(' using tensorflow model')
    from networks.model_tf import * # pylint: disable=W0614

def main(argv):

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    # for reproducing
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3, a.ngf))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    # prepare dataset
    examples = load_batch_examples(a)
    print("examples count = %d" % examples.count)

    # pix2pix model
    out_channels = int(examples.targets.get_shape()[-1])
    pix_model = pix2pix(a, out_channels)

    # inputs and targets are [batch_size, height, width, channels]
    model = pix_model.create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs  = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs  = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs":  tf.map_fn(tf.image.encode_png, converted_inputs,  dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # compute max_steps
    max_steps = 2**32
    if a.max_epochs is not None:
        max_steps = examples.steps_per_epoch * a.max_epochs
    if a.max_steps is not None:
        max_steps = a.max_steps

    # to save checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    # log directory
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

    # use differnt hooks for training and testing
    if a.mode == "test":
        hooks = None
    else:
        # hooks for tf.train.MonitoredTrainingSession
        from networks.custom_hooks import TraceHook, DisplayHook, LoggingTensorHook

        #
        train_hooks = [tf.train.StopAtStepHook(last_step=max_steps),]
        if a.checkpoint:
            train_hooks.append(tf.train.CheckpointSaverHook(
                checkpoint_dir=a.checkpoint,
                save_steps=a.save_freq,
                saver=saver
            ))

        if a.summary_freq:
            train_hooks.append(tf.train.SummarySaverHook(
                save_steps=a.summary_freq,
                output_dir=logdir,
                scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all())
            ))

        if a.progress_freq:
            train_hooks.append(LoggingTensorHook(
                tensors={"discrim_loss": model.discrim_loss,
                         "gen_loss_GAN": model.gen_loss_GAN,
                         "gen_loss_L1": model.gen_loss_L1},
                batch_size=a.batch_size, max_steps=max_steps,
                steps_per_epoch=examples.steps_per_epoch,
                every_n_iter=a.progress_freq,
            ))

        if a.trace_freq:
            train_hooks.append(TraceHook(ckptdir=logdir, every_n_step=a.trace_freq))

        if a.display_freq:
            train_hooks.append(DisplayHook(display_fetches, a.output_dir, every_n_step=a.display_freq))

        #
        hooks = train_hooks

    # don't take the whole memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   #pylint: disable=E1101

    # another way to do it
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as sess:

        print("parameter_count =", sess.run(parameter_count))

        # load previous checkpoint
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for _ in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, a.output_dir)
                for i, f in enumerate(filesets):
                    print("{}: evaluated image: {}".format(i, f["name"]))
                index_path = append_index(filesets, a.output_dir)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        else:

            # training only
            while not sess.should_stop():
                #
                fetches = {
                    "train": model.train,
#                    "global_step": tf.training_util._get_or_create_global_step_read(),    #pylint: disable=protected-access
                }

                # the run
                results = sess.run(fetches)

if __name__ == '__main__':
    # run the main function
    tf.app.run(main)
