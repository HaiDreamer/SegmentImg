import numpy as np, cv2
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance

class AdvancedAugmentation:
    def __init__(self, rotation_range=30, brightness_range=(0.8,1.2),
                 contrast_range=(0.8,1.2), saturation_range=(0.8,1.2),
                 elastic_alpha=100, elastic_sigma=10, elastic_prob=0.5):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

    def _elastic_transform(self, image, mask):
        if np.random.rand() > self.elastic_prob: return image, mask
        shape = image.shape[:2]
        dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        if image.ndim == 3:
            distorted_image = np.zeros_like(image)
            for i in range(image.shape[2]):
                distorted_image[:,:,i] = ndi.map_coordinates(image[:,:,i], indices, order=1, mode='reflect').reshape(shape)
        else:
            distorted_image = ndi.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        distorted_mask = ndi.map_coordinates(mask.astype(np.float32), indices, order=0, mode='reflect').reshape(shape).astype(mask.dtype)
        return distorted_image, distorted_mask

    def _color_jitter(self, image):
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        if self.brightness_range:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(np.random.uniform(*self.brightness_range))
        if self.contrast_range:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(np.random.uniform(*self.contrast_range))
        if self.saturation_range:
            img_pil = ImageEnhance.Color(img_pil).enhance(np.random.uniform(*self.saturation_range))
        return np.array(img_pil).astype(np.float32) / 255.0

    def _random_rotation(self, image, mask):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]; center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine((image * 255).astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
        return img.astype(np.float32) / 255.0, msk.astype(mask.dtype)

    def apply(self, image, mask):
        if self.rotation_range > 0 and np.random.rand() < 0.5:
            image, mask = self._random_rotation(image, mask)
        if np.random.rand() < 0.5:
            image = self._color_jitter(image)
        image, mask = self._elastic_transform(image, mask)
        return image, mask
