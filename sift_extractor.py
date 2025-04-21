import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


class SIFTExtractor:
    def __init__(self, sigma=1.6, k=1.414, contrast_thresh=0.04, edge_thresh=10, magnitude_thresh=0.0):
        self.sigma = sigma
        self.k = k
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh
        self.magnitude_thresh = magnitude_thresh
        self.num_octaves = 4
        self.num_scales = 5

    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)

        gaussian_pyramid = self._generate_gaussian_pyramid(image)
        dog_pyramid = self._generate_dog_pyramid(gaussian_pyramid)
        keypoints = self._detect_keypoints(dog_pyramid)
        keypoints = self._localize_keypoints(dog_pyramid, keypoints)
        keypoints = self._assign_orientation(gaussian_pyramid, keypoints)
        descriptors = self._compute_descriptors(gaussian_pyramid, keypoints)
        return keypoints, descriptors

    def _generate_gaussian_pyramid(self, image):
        octaves = []
        base = image.copy()
        for o in range(self.num_octaves):
            scales = []
            for s in range(self.num_scales):
                sigma = self.sigma * (self.k ** s)
                blurred = gaussian_filter(base, sigma)
                scales.append(blurred)
            octaves.append(scales)
            base = cv2.resize(base, (base.shape[1] // 2, base.shape[0] // 2))
        return octaves

    def _generate_dog_pyramid(self, gaussian_pyramid):
        dog_pyramid = []
        for octave in gaussian_pyramid:
            dogs = [octave[i+1] - octave[i] for i in range(len(octave)-1)]
            dog_pyramid.append(dogs)
        return dog_pyramid

    def _detect_keypoints(self, dog_pyramid):
        keypoints = []
        for o_idx, dogs in enumerate(dog_pyramid):
            for s in range(1, len(dogs)-1):
                prev_img, curr_img, next_img = dogs[s-1], dogs[s], dogs[s+1]
                h, w = curr_img.shape
                for y in range(1, h-1):
                    for x in range(1, w-1):
                        val = curr_img[y, x]
                        if abs(val) < self.contrast_thresh:
                            continue
                        patch = np.stack([
                            prev_img[y-1:y+2, x-1:x+2],
                            curr_img[y-1:y+2, x-1:x+2],
                            next_img[y-1:y+2, x-1:x+2]
                        ])
                        if val == patch.max() or val == patch.min():
                            keypoints.append((o_idx, s, y, x))
        return keypoints

    def _localize_keypoints(self, dog_pyramid, keypoints):
        refined = []
        for o, s, y, x in keypoints:
            dog = dog_pyramid[o][s]
            if x < 1 or x >= dog.shape[1]-1 or y < 1 or y >= dog.shape[0]-1:
                continue
            dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]
            dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]
            dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0
            det = dxx * dyy - dxy ** 2
            if det <= 0:
                continue
            tr = dxx + dyy
            r = (tr ** 2) / det
            if r < ((self.edge_thresh + 1) ** 2) / self.edge_thresh:
                refined.append((o, s, y, x))
        return refined

    def _assign_orientation(self, gaussian_pyramid, keypoints):
        oriented = []
        for o, s, y, x in keypoints:
            img = gaussian_pyramid[o][s]
            if x < 1 or y < 1 or x >= img.shape[1] - 1 or y >= img.shape[0] - 1:
                continue
            dx = img[y, x+1] - img[y, x-1]
            dy = img[y-1, x] - img[y+1, x]
            mag = np.sqrt(dx**2 + dy**2)
            if mag < self.magnitude_thresh:
                continue
            angle = np.arctan2(dy, dx)
            scale = self.sigma * (self.k ** s) * (2 ** o)
            oriented.append(((o, s, y, x, scale), angle))
        return oriented

    def _compute_descriptors(self, gaussian_pyramid, keypoints):
        descriptors = []
        for (o, s, y, x, scale), theta in keypoints:
            img = gaussian_pyramid[o][s]
            patch_size = int(round(16 * scale))
            if patch_size < 16:
                patch_size = 16

            # Create rotation matrix
            M = cv2.getRotationMatrix2D((x, y), np.degrees(-theta), 1.0)
            rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

            # Crop patch around (x, y)
            half = 8
            cx, cy = int(x), int(y)
            if cy - half < 0 or cy + half >= rotated.shape[0] or cx - half < 0 or cx + half >= rotated.shape[1]:
                continue

            patch = rotated[cy - half:cy + half, cx - half:cx + half]

            # Compute gradients
            dx = patch[:, 1:] - patch[:, :-1]
            dy = patch[1:, :] - patch[:-1, :]

            magnitude = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
            angle = np.arctan2(dy[:, :-1], dx[:-1, :])

            if magnitude.shape != (15, 15):
                continue

            weight = cv2.getGaussianKernel(15, 1.5)
            weight = weight @ weight.T  # Make 2D Gaussian
            magnitude *= weight         # Weight the magnitudes

            bins = np.zeros((4, 4, 8))
            for i in range(4):
                for j in range(4):
                    for u in range(4):
                        for v in range(4):
                            yy = i * 4 + u
                            xx = j * 4 + v
                            if yy >= magnitude.shape[0] or xx >= magnitude.shape[1]:
                             continue  # Skip out-of-bounds
                            m = magnitude[yy, xx]
                            a = angle[yy, xx]
                            bin_idx = int(((a + np.pi) % (2 * np.pi)) / (2 * np.pi) * 8)
                            bins[i, j, bin_idx] += m

            descriptor = bins.flatten()
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            descriptor = np.clip(descriptor, 0, 0.2)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm

            descriptors.append(descriptor)

        return np.array(descriptors, dtype=np.float32)

def match_features(desc1, desc2, ratio_thresh=0.75):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        if len(distances) < 2:
            continue
        nearest = np.argsort(distances)
        if distances[nearest[0]] < ratio_thresh * distances[nearest[1]]:
            matches.append((i, nearest[0]))
    return matches


def draw_matches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    target_h = max(h1, h2)
    target_w = max(w1, w2)
    img1 = cv2.resize(img1, (target_w, target_h))
    img2 = cv2.resize(img2, (target_w, target_h))

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    canvas = np.vstack((img1_color, img2_color))

    for i1, i2 in matches:
        _, _, y1, x1, _ = kp1[i1][0]
        _, _, y2, x2, _ = kp2[i2][0]
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2 + target_h))
        cv2.circle(canvas, pt1, 4, (0, 255, 0), 1)
        cv2.circle(canvas, pt2, 4, (0, 255, 0), 1)
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

    return canvas


# import numpy as np
# import cv2
# from scipy.ndimage import gaussian_filter


# class SIFTExtractor:
#     def __init__(self, sigma=1.6, k=1.414, contrast_thresh=0.04, edge_thresh=10, magnitude_thresh=0.0):
#         self.sigma = sigma
#         self.k = k
#         self.contrast_thresh = contrast_thresh
#         self.edge_thresh = edge_thresh
#         self.magnitude_thresh = magnitude_thresh
#         self.num_octaves = 4
#         self.num_scales = 5

#     def extract(self, image):
#         gaussian_pyramid = self._generate_gaussian_pyramid(image)
#         dog_pyramid = self._generate_dog_pyramid(gaussian_pyramid)
#         keypoints = self._detect_keypoints(dog_pyramid)
#         keypoints = self._localize_keypoints(dog_pyramid, keypoints)
#         keypoints = self._assign_orientation(gaussian_pyramid, keypoints)
#         descriptors = self._compute_descriptors(gaussian_pyramid, keypoints)
#         return keypoints, descriptors

#     def _generate_gaussian_pyramid(self, image):
#         octaves = []
#         base = image.astype(np.float32)
#         for o in range(self.num_octaves):
#             scales = []
#             for s in range(self.num_scales):
#                 sigma = self.sigma * (self.k ** s)
#                 blurred = gaussian_filter(base, sigma)
#                 scales.append(blurred)
#             octaves.append(scales)
#             base = base[::2, ::2]  # Downsample by 2
#         return octaves

#     def _generate_dog_pyramid(self, gaussian_pyramid):
#         dog_pyramid = []
#         for octave in gaussian_pyramid:
#             dogs = []
#             for i in range(1, len(octave)):
#                 dog = octave[i] - octave[i - 1]
#                 dogs.append(dog)
#             dog_pyramid.append(dogs)
#         return dog_pyramid

#     def _detect_keypoints(self, dog_pyramid):
#         keypoints = []
#         for o_idx, dogs in enumerate(dog_pyramid):
#             for s in range(1, len(dogs) - 1):
#                 prev_img = dogs[s - 1]
#                 curr_img = dogs[s]
#                 next_img = dogs[s + 1]
#                 h, w = curr_img.shape
#                 for y in range(1, h - 1):
#                     for x in range(1, w - 1):
#                         patch = np.stack([
#                             prev_img[y - 1:y + 2, x - 1:x + 2],
#                             curr_img[y - 1:y + 2, x - 1:x + 2],
#                             next_img[y - 1:y + 2, x - 1:x + 2]
#                         ])
#                         val = curr_img[y, x]
#                         if abs(val) < self.contrast_thresh:
#                             continue
#                         if val == patch.max() or val == patch.min():
#                             keypoints.append((o_idx, s, y, x))
#         return keypoints

#     def _localize_keypoints(self, dog_pyramid, keypoints):
#         refined_keypoints = []
#         for kp in keypoints:
#             o, s, y, x = kp
#             dog_img = dog_pyramid[o][s]
#             h, w = dog_img.shape
#             if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
#                 continue

#             dx = (dog_img[y, x + 1] - dog_img[y, x - 1]) / 2.0
#             dy = (dog_img[y + 1, x] - dog_img[y - 1, x]) / 2.0
#             dxx = dog_img[y, x + 1] + dog_img[y, x - 1] - 2 * dog_img[y, x]
#             dyy = dog_img[y + 1, x] + dog_img[y - 1, x] - 2 * dog_img[y, x]
#             dxy = (dog_img[y + 1, x + 1] - dog_img[y + 1, x - 1] - dog_img[y - 1, x + 1] + dog_img[y - 1, x - 1]) / 4.0

#             trace = dxx + dyy
#             det = dxx * dyy - dxy ** 2
#             if det <= 0:
#                 continue

#             r = (trace ** 2) / det
#             if r < ((self.edge_thresh + 1) ** 2) / self.edge_thresh:
#                 refined_keypoints.append((o, s, y, x))

#         return refined_keypoints

#     def _assign_orientation(self, gaussian_pyramid, keypoints):
#         oriented_keypoints = []
#         for kp in keypoints:
#             o, s, y, x = kp
#             image = gaussian_pyramid[o][s]
#             if x < 1 or x >= image.shape[1] - 1 or y < 1 or y >= image.shape[0] - 1:
#                 continue
#             dx = image[y, x + 1] - image[y, x - 1]
#             dy = image[y - 1, x] - image[y + 1, x]
#             magnitude = np.sqrt(dx ** 2 + dy ** 2)
#             if magnitude < self.magnitude_thresh:
#                 continue
#             orientation = np.arctan2(dy, dx)
#             oriented_keypoints.append(((o, s, y, x), orientation))
#         return oriented_keypoints

#     def _compute_descriptors(self, gaussian_pyramid, keypoints):
#         descriptors = []
#         for (o, s, y, x), orientation in keypoints:
#             image = gaussian_pyramid[o][s]
#             if y < 8 or x < 8 or y + 8 >= image.shape[0] or x + 8 >= image.shape[1]:
#                 continue

#             patch = image[y - 8:y + 8, x - 8:x + 8]
#             grad_x = patch[:, 1:] - patch[:, :-1]
#             grad_y = patch[1:, :] - patch[:-1, :]

#             magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
#             angle = np.arctan2(grad_y[:, :-1], grad_x[:-1, :]) - orientation

#             if magnitude.shape != (15, 15) or angle.shape != (15, 15):
#                 continue  # Skip corrupted patches

#             bins = np.zeros((4, 4, 8))
#             for i in range(4):
#                 for j in range(4):
#                     for u in range(4):
#                         for v in range(4):
#                             yy = i * 4 + u
#                             xx = j * 4 + v
#                             if yy >= 15 or xx >= 15:
#                                 continue
#                             m = magnitude[yy, xx]
#                             a = angle[yy, xx]
#                             bin_idx = int(((a + np.pi) % (2 * np.pi)) / (2 * np.pi) * 8)
#                             bins[i, j, bin_idx] += m

#             descriptor = bins.flatten()
#             norm = np.linalg.norm(descriptor)
#             if norm > 0:
#                 descriptor = descriptor / norm
#             descriptor = np.clip(descriptor, 0, 0.2)
#             norm = np.linalg.norm(descriptor)
#             if norm > 0:
#                 descriptor = descriptor / norm

#             descriptors.append(descriptor)
#         return np.array(descriptors, dtype=np.float32)


# def match_features(desc1, desc2):
#     matches = []
#     for i, d1 in enumerate(desc1):
#         distances = np.linalg.norm(desc2 - d1, axis=1)
#         best_idx = np.argmin(distances)
#         matches.append((i, best_idx))
#     return matches


# def draw_matches(img1, kp1, img2, kp2, matches):
#     target_height = 600
#     target_width = 600

#     # Resize both images
#     img1_resized = cv2.resize(img1, (target_width, target_height))
#     img2_resized = cv2.resize(img2, (target_width, target_height))

#     img1_color = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
#     img2_color = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)

#     canvas = np.zeros((2 * target_height, target_width, 3), dtype=np.uint8)
#     canvas[:target_height] = img1_color
#     canvas[target_height:] = img2_color

#     for i1, i2 in matches:
#         _, _, y1, x1 = kp1[i1][0]
#         _, _, y2, x2 = kp2[i2][0]

#         scale_y1 = target_height / img1.shape[0]
#         scale_x1 = target_width / img1.shape[1]
#         scale_y2 = target_height / img2.shape[0]
#         scale_x2 = target_width / img2.shape[1]

#         pt1 = (int(x1 * scale_x1), int(y1 * scale_y1))
#         pt2 = (int(x2 * scale_x2), int(y2 * scale_y2) + target_height)

#         cv2.circle(canvas, pt1, 4, (0, 0, 255), 1)
#         cv2.circle(canvas, pt2, 4, (0, 0, 255), 1)
#         cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

#     return canvas

