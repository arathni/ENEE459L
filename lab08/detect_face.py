import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    confidence_scores = box_scores[:, 4]
    coordinate_tuples = box_scores[:, :4]
    indices = np.argsort(confidence_scores)
    candidates = []
    for i in range(len(indices)):
        if i >= len(indices) - candidate_size:
            candidates.append(indices[i])
    indices = np.array(candidates, dtype=int)
    picked = []

    while len(indices) > 0:
        current = indices[-1]
        picked.append(current)
        if len(picked) == top_k or len(indices) == 1:
            break

        indices = indices[:-1]
        current2 = coordinate_tuples[current]
        keep = []

        for index in indices:
            remaining = coordinate_tuples[index]
            x1 = max(current2[0], remaining[0])
            y1 = max(current2[1], remaining[1])
            x2 = min(current2[2], remaining[2])
            y2 = min(current2[3], remaining[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            width = remaining[2] - remaining[0]
            height = remaining[3] - remaining[1]
            area = (current2[2] - current2[0]) * (current2[3] - current2[1])
            union = area + width * height - intersection
            iou = intersection / union
            if iou <= iou_threshold:
                keep.append(index)

        indices = np.array(keep, dtype=int)

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    data = np.array(image.resize((320, 240)), dtype=np.float32)
    data = (data - 127.0) / 128.0
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options,
                                   providers=available_providers)
    scores, boxes = session.run(None, {session.get_inputs()[0].name: data})

    scores = scores[0]
    boxes = boxes[0]
    confidence_scores = scores[:, 1]
    keep = []
    for i in range(len(confidence_scores)):
        if confidence_scores[i] >= conf_threshold:
            keep.append(i)
    boxes = boxes[keep]
    confidence_scores = confidence_scores[keep]

    box_scores = np.column_stack((boxes, confidence_scores))
    box_scores = hard_nms(box_scores, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for box in box_scores:
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, y1 - 20), f"Face: {box[4]:.2f}", fill="lime")
    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    out = np.zeros((height, width), dtype=np.float32)
    angle = np.degrees(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                q = magnitude[i, j - 1]
                r = magnitude[i, j + 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                out[i, j] = magnitude[i, j]
            else:
                out[i, j] = 0

    return out

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong = 255
    weak = 50
    suppressed = 0
    out = np.zeros(img.shape, dtype=np.uint8)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] >= high_thresh:
                out[i, j] = strong
            elif img[i, j] >= low_thresh:
                out[i, j] = weak
            else:
                out[i, j] = suppressed

    weak_pixels = []
    for i in range(height):
        for j in range(width):
            if out[i, j] == weak:
                weak_pixels.append((i, j))

    for i, j in weak_pixels:
        found_strong = False
        for row in range(i - 1, i + 2):
            for col in range(j - 1, j + 2):
                if 0 <= row < height and 0 <= col < width:
                    if out[row, col] == strong:
                        found_strong = True

        if found_strong:
            out[i, j] = strong
        else:
            out[i, j] = suppressed

    return out

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path)
    image = image.convert("L")
    data = np.array(image)
    data = data.astype(np.float32)
    smoothed = gaussian_filter(data, sigma=1.4)

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    Gx = convolve(smoothed, Kx)
    Gy = convolve(smoothed, Ky)
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    M = M / M.max() * 255
    angle = np.arctan2(Gy, Gx)

    thinned = non_maximum_suppression(M, angle)
    edges = hysteresis_thresholding(thinned, low, high)

    edges = edges.astype(np.uint8)
    output = Image.fromarray(edges)
    output.save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")
