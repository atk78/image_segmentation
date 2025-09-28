import matplotlib.pyplot as plt


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.title(), fontsize=20)
        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
    plt.show()


def metrics_plot(history, save_file_path=None):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["valid_loss"], label="valid loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_iou_score"], label="train IoU score")
    plt.plot(history["valid_iou_score"], label="valid IoU score")
    plt.xlabel("Epochs")
    plt.ylabel("IoU score")
    plt.legend()
    plt.grid()

    if save_file_path is not None:
        plt.savefig(save_file_path)
    else:
        plt.show()
