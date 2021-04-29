from matplotlib.pyplot import subplots, savefig, close
from os import listdir
from os.path import exists
from numpy import loadtxt

overwrite = False

if __name__ == "__main__":
    data_directory = "build/saves/"
    saves_directory = "snapshots/"
    for frame in listdir(data_directory):
        img_path = saves_directory + frame.replace(".dat", ".png")
        if exists(img_path) and not overwrite:
            continue
        else:
            agents = loadtxt(data_directory + frame)
            fig, ax = subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(agents, cmap="copper")
            savefig(img_path)
            close(fig)
