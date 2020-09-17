from PIL import Image
from torchvision import transforms
import numpy as np

def transform(image1, image2):
    # Performs preproocessing of atari environment as described in:
    # Human-level control through deep reinforcement learning by Mnih et al.

    # Take max pixel values between the 2 frames
    image_in = np.maximum(image1, image2)
    # To PIL image
    image_out = Image.fromarray(image_in)
    # RGB -> Luminosity (Grayscale)
    image_out = image_out.convert("L")
    # Resize
    image_out = image_out.resize((84, 84))
    # Back to Tensor
    return transforms.functional.to_tensor(image_out)


class LRScheduler():
    def __init__(self, args):
        self.gamma = (1e-10 / args.lr) ** (args.update_freq * args.num_procs / args.steps)
        self.lr = args.lr

    def step(self):
        self.lr *= self.gamma
        return self.lr
