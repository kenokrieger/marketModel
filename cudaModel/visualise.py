from matplotlib.pyplot import subplots, savefig, close
from os import listdir, remove, stat
from os.path import exists
from numpy import loadtxt
import time

overwrite = False
data_directory = "build/saves/"
saves_directory = "snapshots/"
plot_multiple = False
frame_number = 563850

def save_color_map(data_directory, saves_directory):
    for frame in listdir(data_directory):
        img_path = saves_directory + frame.replace(".dat", ".png")

        if exists(img_path) and not overwrite:
            continue

        data_file_path = data_directory + frame
        if stat(data_file_path).st_size < 250e6:
            continue

        print("Plotting {}".format(frame))
        agents = loadtxt(data_file_path)
        fig, ax = subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(agents, cmap="copper")
        savefig(img_path)
        close(fig)
        remove(data_file_path)


def save_frame(data_directory, saves_directory, frame_number):
    frame = "snapshot.dat" #"frame_{}.dat".format(frame_number)
    img_path = saves_directory + frame.replace(".dat", ".png")
    data_file_path = data_directory + frame
    print("Plotting {}".format(frame))
    agents = loadtxt(data_file_path)
    fig, ax = subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(agents, cmap="copper")
    savefig(img_path)
    close(fig)


if __name__ == "__main__":
    save_frame("", saves_directory, 1)
    exit()
    if not plot_multiple:
        save_frame(data_directory, saves_directory, frame_number)
    else:
        while True:
            save_color_map(data_directory, saves_directory)
