import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np


def ax2bgr(ax):
    canvas = ax.figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return img


def data2img(data, size=500, color=True):
    img = process_img(data, smooth=False)
    h, w = img.shape
    # opencv: change (w, h) here
    img = cv2.resize(img, (int(size / h * w), size))
    img = (img * 255).astype(np.uint8)
    if color:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def process_img(img, smooth=True, eps=1e-16):
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + eps)
    if smooth:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1, sigmaY=0)
    return img


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


def render_single(y, bottom=None, top=None):
    fig, ax = init_fig()
    ax.plot(y, '.r')
    ax.set_ylim(bottom=bottom, top=top)
    res = ax2bgr(ax)
    plt.close(fig)
    return res


def render_double(y1, y2, label1=None, label2=None, bottom=None, top=None):
    fig, ax = init_fig()
    ax.plot(y1, '.b', label=label1)
    ax.plot(y2, '.r', label=label2)
    ax.set_ylim(bottom=bottom, top=top)
    ax.legend()
    res = ax2bgr(ax)
    plt.close(fig)
    return res


def render_env(ax, xs, x_preds=None, rs=None):
    # xs
    xs = np.array(xs)
    ax.plot(xs[:, 0], xs[:, 1], 'b', label='pos')
    ax.plot(xs[-1, 0], xs[-1, 1], 'xb')

    # x_preds
    if x_preds is not None:
        x_preds = np.array(x_preds)
        ax.plot(x_preds[:, 0], x_preds[:, 1], 'r', label='pred')
        ax.plot(x_preds[-1, 0], x_preds[-1, 1], 'xr')

        error = l2_distance(x_preds[-1, :], xs[-1, :])
        ax.set_title('error={:.2f}'.format(error))

    # rs
    if rs is not None:
        rs = np.array(rs)[:, 0, 5]
        idx = (rs > 0)
        ax.plot(xs[idx, 0], xs[idx, 1], 'or')

    ax.legend(loc='lower right')
    res = ax2bgr(ax)
    return res
