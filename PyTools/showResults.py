import numpy as np
import argparse


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Show results for different parameter dG estimation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    return  args.v


if __name__ == "__main__":
    verbose = parse_arguments()
    iterations = [(25, 100), (50, 100), (100, 100), (200, 100), (400, 100), 
                  (25, 200), (50, 200), (100, 200), (200, 200), (400, 200), 
                  (25, 400), (50, 400), (100, 400), (200, 400), (400, 400)]
    for k, cl in iterations:
        try:
            dG = np.loadtxt("%dlag/%dcl/results.txt" % (k, cl))
        except:
            continue
        if verbose:
            print k, cl, " +- ".join(map(str, dG[1:3]))
        else:
            print " +- ".join(map(str, dG[1:3]))
