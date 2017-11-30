import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def sort_split_by_numbers(traj_name):
    trajNum, snapNum = traj_name.split("_")[-2:]
    return (int(trajNum), int(snapNum[:-4]))


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Write certain conformations specified from a COM_BE.py pdb"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-t", "--TIC", type=int, nargs="+", help="TICA index for x axis")
    parser.add_argument("-m", "--metricCol", type=int, nargs="+", help="Column of the metrics to consider")
    parser.add_argument("-f", "--folder", type=str, default="", help="Path to the report files")
    args = parser.parse_args()
    return args.TIC, args.metricCol, args.folder


TIC, metricCol, report_folder = parse_arguments()
nTics = len(TIC)
if report_folder and report_folder[-1] != "/":
    report_folder += "/"
data_folder = "tica_COM/"
COM_tica = []
files = glob.glob(data_folder+"traj_*")
files.sort(key=sort_split_by_numbers)
plots_folder = "tica_scatter/"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

data = []
for traj in files:
    traj_num, snap_num = traj.split("_")[-2:]
    traj_data = np.loadtxt(traj)
    if len(traj_data.shape) < 2:
        traj_data = traj_data[np.newaxis, :]
    report_data = np.loadtxt(report_folder+"%s/report_%s" % (traj_num, snap_num[:-4]))
    if len(report_data.shape) < 2:
        report_data = report_data[np.newaxis, :]
    data.extend(np.vstack((traj_data[:, [tic+2 for tic in TIC]].T, report_data[:, metricCol].T)).T.tolist())

data = np.array(data)

for iTic, tic in enumerate(TIC):
    for iMetric, metric in enumerate(metricCol):
        plt.figure()
        plt.scatter(data[:, iTic], data[:, iMetric+nTics])
        plt.xlabel("TICA %d" % tic)
        plt.ylabel("Metric column %d" % metric)
        plt.savefig(plots_folder+"scatter_TIC%d_metric_%d.png" % (tic, metric))

plt.show()
