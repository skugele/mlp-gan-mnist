import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from tensorflow.examples.tutorials import mnist


def get_mnist_datasets(data_dir='/tmp'):
  return mnist.input_data.read_data_sets(data_dir, one_hot=True)


def get_image(values, width, height):
  """Converts a flattened (1d) array of pixel intensities into a 2d image

  Args:
      values: A 1d ndarray containing pixel intensities. 
      width: The width of the resulting image.
      height: The height of the resulting image.

  Returns:
      A 2d ndarray containing the provided pixel intensities with the specified
      width and height.
  """
  return (255 * values.reshape(width, height)).astype(np.uint8)


def get_label(label):
  """Converts a one-hot encoded digit label into a numeric (human-readable) label

  Args:
      label: A 1d ndarray containing a one-hot encoded label value.
    
  Returns:
      A string with the label value.  (For example, "1")
  """
  return list(label).index(1)


def display_image(image):
  plt.imshow(image, cmap='Greys_r')


def write_image(image, filepath):
  imwrite(filepath, image)


def write_images_plot(images, img_width, img_height, rows, cols, filepath):
  fig = plt.figure(dpi=120)
  fig.set_size_inches(6, 6)
  gs = gridspec.GridSpec(rows, cols)
  gs.update(wspace=0.1, hspace=0.1)

  for i, img in enumerate(images):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(img.reshape(img_width, img_height), cmap='Greys_r')

  plt.savefig(filepath, bbox_inches='tight')
  plt.close(fig)
  return fig
