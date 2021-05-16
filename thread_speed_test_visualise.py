from sys import argv
from os import mkdir
from os.path import join, exists

from matplotlib.pyplot import subplots, savefig, close
from numpy import loadtxt

saves_directory = "images/"


def save_frame(frame, saves_directory):

    headers = []
    with open(frame, 'r') as f:
        header = f.readline()
        while header[0] == '#':
            headers.append(header.replace('#', '').replace('\n', ''))
            header = f.readline()
    img_name = "speed test with " + ", ".join(headers[:-1])

    thread_speed = loadtxt(frame)
    img_path = join(saves_directory, img_name + ".png")
    print("Plotting {}".format(img_name))

    fig, ax = subplots(figsize=(12, 8))
    fig.suptitle(img_name)
    ax.plot(thread_speed[:, 0], thread_speed[:, 1])
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Spin updates per nano second")
    savefig(img_path)
    close(fig)


if __name__ == "__main__":
    if not exists(saves_directory):
        mkdir(saves_directory)
    for file in argv[1].split(".dat"):
        if not file:
            continue
        else:
            file += ".dat"
        save_frame(file, saves_directory)
