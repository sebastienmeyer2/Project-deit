"""Utilitary functions and command lines to plot scores."""


import os

import json
from pathlib import Path

from typing import List

import argparse

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


import utils as ut


plt.style.use("seaborn-darkgrid")
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.prop_cycle"] = cycler("color", plt.get_cmap("Dark2")(np.linspace(0, 1, 5)))


def get_args_parser():
    """Parser getter.

    Returns
    -------
    parser : `argparse.ArgumentParser`
        Parser of args.
    """
    parser = argparse.ArgumentParser("DeiT plotting script", add_help=False)

    parser.add_argument(
        "--metric",
        default="loss",
        type=str,
        help="""Name of the metric to compare. Default: "loss"."""
    )

    parser.add_argument(
        "--models-folders",
        default="",
        type=lambda s: [x for x in s.split(",")],
        help="""Paths to the results for chosen models. Default: ""."""
    )
    parser.add_argument(
        "--output-folder",
        default="",
        help="""Path where to save, empty for no saving. Default: ""."""
    )

    parser.add_argument(
        "--model-file",
        default="log.txt",
        type=str,
        help="""Name of the file where results are stored. Default: "log.txt"."""
    )
    parser.add_argument(
        "--train",
        default="False",
        type=str,
        help="""
             If "True", will plot and save results for the training, otherwise for test.
             Default: "False".
             """
    )

    return parser


def compare_metric(
    metric: str, models_folders: List[str], output_folder: str, model_file: str = "log.txt",
    train: bool = False
):
    """Plot and save a comparison of models scores.

    Parameters
    ----------
    metric : str
        Name of the metric to compare.

    models_folders : list of str
        List of names of directories where results for each model are stores.

    output_folder : str
        Name of the directory where to save comparison.

    model_file : str, default="log.txt"
        Name of the file where results are stored.

    train : bool, default=False
        If "True", will plot and save results for the training, otherwise for test.
    """
    metric_full = ("train_" if train else "test_") + metric
    res = {}

    for model_folder in models_folders:

        # Retrieve log file
        if not os.path.exists(model_folder):
            raise ValueError(f"Model folder {model_folder} is missing.")

        path_to_res = model_folder + model_file

        model_name = model_folder.split("/")[-2]
        res[model_name] = {"epoch": [], metric_full: []}

        # Read results
        with open(path_to_res, encoding="utf-8") as f:

            for line in f:

                values = json.loads(line)

                if metric_full in values:

                    res[model_name]["epoch"].append(values["epoch"])
                    res[model_name][metric_full].append(values[metric_full])

    # Plotting and saving
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    path_to_file = output_folder + "comp.png"

    for model_folder in models_folders:

        model_name = model_folder.split("/")[-2]

        plt.plot(res[model_name]["epoch"], res[model_name][metric_full], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel(" ".join(metric_full.split("_")).capitalize())
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("DeiT plotting script.", parents=[get_args_parser()])
    args = parser.parse_args()

    # Retrieve parameters
    metric = args.metric
    models_folders = args.models_folders
    output_folder = args.output_folder
    model_file = args.model_file
    train = ut.str2bool(args.train)

    compare_metric(metric, models_folders, output_folder, model_file=model_file, train=train)
