import glob
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
    parser.add_argument("TICx", type=int, help="TICA index for x axis")
    parser.add_argument("TICy", type=int, help="TICA index for y ayis")
    parser.add_argument("metricCol", type=int, help="Column of the metric to consider")
    parser.add_argument("-f", "--folder", type=str, default="", help="Path to the report files")
    args = parser.parse_args()
    return args.TICx, args.TICy, args.metricCol, args.folder


TICx, TICy, metricCol, report_folder = parse_arguments()
if report_folder and report_folder[-1] != "/":
    report_folder += "/"
data_folder = "tica_COM/"
COM_tica = []
files = glob.glob(data_folder+"traj_*")
files.sort(key=sort_split_by_numbers)

data = []
for traj in files:
    traj_num, snap_num = traj.split("_")[-2:]
    traj_data = np.loadtxt(traj)
    if len(traj_data.shape) < 2:
        traj_data = traj_data[np.newaxis, :]
    report_data = np.loadtxt(report_folder+"%s/report_%s" % (traj_num, snap_num[:-4]))
    if len(report_data.shape) < 2:
        report_data = report_data[np.newaxis, :]
    data.extend(np.vstack((traj_data[:, 2+TICx].T, traj_data[:, 2+TICy].T, report_data[:, metricCol].T)).T.tolist())

data = np.array(data)
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])

plt.xlabel("TICA %d" % TICx)
plt.ylabel("TICA %d" % TICy)
plt.colorbar()
plt.savefig("scatter_TIC%d_TIC%d.png" % (TICx, TICy))

plt.show()
