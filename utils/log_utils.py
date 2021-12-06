import errno
import os
import sys


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_folders(config):
    if not os.path.exists(config.DEBUG):
        os.mkdir(config.DEBUG)
        os.mkdir('{}/checkpoints'.format(config.DEBUG))
        os.mkdir('{}/images'.format(config.DEBUG))
    if not os.path.exists('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT))
    if not os.path.exists('{}/images/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/images/{}'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/train'.format(config.DEBUG, config.EXPERIMENT))
        os.mkdir('{}/images/{}/test'.format(config.DEBUG, config.EXPERIMENT))
        # os.mkdir('{}/images/{}/train/colormaps'.format(config.DEBUG, config.EXPERIMENT))
        # os.mkdir('{}/images/{}/test/colormaps'.format(config.DEBUG, config.EXPERIMENT))
        # os.mkdir('{}/images/{}/train/show_boxes'.format(config.DEBUG, config.EXPERIMENT))
        # os.mkdir('{}/images/{}/test/show_boxes'.format(config.DEBUG, config.EXPERIMENT))
    if not os.path.exists('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT))


# Plots a line-by-line description of a PyTorch model
def model_info(model):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))
