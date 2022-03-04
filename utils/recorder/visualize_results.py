import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tv_tf
from torchvision.utils import make_grid

matplotlib.use("agg")


@torch.no_grad()
def plot_results(data_container, save_path=None):
    """Plot the results conresponding to the batched images based on the `make_grid` method from `torchvision`.

    Args:
        data_container (dict): Dict containing data you want to plot.
        save_path (str): Path of the exported image.
    """
    axes = plt.subplots(nrows=len(data_container), ncols=1)[1].ravel()
    plt.subplots_adjust(hspace=0.03, left=0.05, bottom=0.01, right=0.99, top=0.99)

    for subplot_id, (name, data) in enumerate(data_container.items()):
        grid = make_grid(data, nrow=data.shape[0], padding=2, normalize=False)
        grid_image = np.asarray(tv_tf.to_pil_image(grid))
        axes[subplot_id].imshow(grid_image)
        axes[subplot_id].set_ylabel(name)
        axes[subplot_id].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    matplotlib.use("tkagg")
    data_container = dict(
        image=torch.randn(4, 1, 320, 320),
        mask=torch.randn(4, 1, 320, 320),
        prediction=torch.rand(4, 1, 320, 320),
    )
    plot_results(data_container=data_container, save_path=None)
