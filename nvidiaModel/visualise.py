from matplotlib.pyplot import subplots, savefig, close
from os import listdir
from numpy import loadtxt


if __name__ == "__main__":
    data_directory = "build/saves/"
    saves_directory = "snapshots/"
    for frame in listdir(data_directory):
        agents = loadtxt(data_directory + frame)
        fig, ax = subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(agents, cmap="gray")
        savefig(saves_directory + "frame{}.png".format(frame.replace("frame_", "").replace(".dat", "")))
        close(fig)
