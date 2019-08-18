import os
import numpy as np
from matplotlib import pyplot as plt

import os
import time
import logging
import operator
import numpy as np
import coloredlogs
from PIL import Image

plt.switch_backend("Agg")


class MetricSaver(object):
    def __init__(self,
                 name,
                 path,
                 ma_weight=0.9,
                 figsize=(6, 4),
                 postfix=".png",
                 save_on_update=True):
        self.name = name
        self.path = path
        self.ma_weight = ma_weight

        self.step = [0.]
        self.original_value = []
        self.value = [0.]
        self.ma_value = [0.]

        self.figsize = figsize
        self.postfix = postfix
        self.save_on_update = save_on_update

    def _test_valid_step(self):
        for i in range(1, len(self.step)):
            if self.step[i] is None:
                return False
            if self.step[i] < self.step[i - 1]:
                return False
        return True

    def update(self, step=None, value=None, save=True):
        self.step.append(step)
        self.original_value.append(value)

        value = np.mean(value)
        self.value.append(value)
        if len(self.ma_value) == 1:
            self.ma_value.append(value)
        else:
            self.ma_value.append(self.ma_value[-1] * self.ma_weight + value *
                                 (1. - self.ma_weight))

        if save is True or self.save_on_update is True:
            self.save()

    def save(self):
        if self._test_valid_step() is True:
            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.step[1:],
                self.value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, self.name + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)

            # fig = plt.figure(figsize=self.figsize)
            # plt.plot(
            #     self.step[1:],
            #     self.ma_value[1:],
            #     '-',
            # )
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(os.path.join(self.path,
            #                          self.name + "_ma" + self.postfix),
            #             bbox_inches='tight')
            # plt.close(fig)
        else:
            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, self.name + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)

        #     fig = plt.figure(figsize=self.figsize)
        #     plt.plot(
        #         self.ma_value[1:],
        #         '-',
        #     )
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(self.path,
        #                              self.name + "_ma" + self.postfix),
        #                 bbox_inches='tight')
        #     plt.close(fig)
        # np.savez(os.path.join(self.path, self.name + ".npz"),
        #          value=self.value[1:],
        #          ma_value=self.ma_value,
        #          original_value=self.original_value)


def build_logger(folder=None, args=None, logger_name=None):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    if folder is not None:
        fh = logging.FileHandler(filename=os.path.join(
            folder, "logfile{}.log".format(time.strftime("%m-%d"))))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s|%(message)s",
                                      "%H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(level=logging.DEBUG,
                        fmt=FORMAT,
                        datefmt=DATEF,
                        level_styles=LEVEL_STYLES)

    def get_list_name(obj):
        if type(obj) is list:
            for i in range(len(obj)):
                if callable(obj[i]):
                    obj[i] = obj[i].__name__
        elif callable(obj):
            obj = obj.__name__
        return obj

    if args is None:
        return logger

    if isinstance(args, dict) is not True:
        args = vars(args)

    sorted_list = sorted(args.items(), key=operator.itemgetter(0))
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    for name, val in sorted_list:
        logger.info("# " + ("%20s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger
