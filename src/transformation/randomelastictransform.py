from torchvision import transforms
import random

class RandomElasticTransform:
    def __init__(self, alpha_range=(0.0, 100.0), sigma=9.0, fill=255):
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.fill = fill

    def __call__(self, img):
        alpha = random.uniform(*self.alpha_range)

        return transforms.ElasticTransform(
            alpha=alpha,
            sigma=self.sigma,
            fill=self.fill
        )(img)
