"""
Interactive HSV Color Finder

Usage:
	python -m car_training.car_training.tasks.manager_based.car_training.mdp.utils.hsv_color_finder --image /path/to/image.jpg

Features:
 - Live sliders for H/S/V min/max thresholds
 - Preview of mask and masked result
 - Optional blur and morphological closing to refine mask
 - Draw a rectangle ROI on the image, then press 'a' to auto-estimate thresholds
 - Press 's' to save mask and parameters next to the image
 - Press 'r' to reset sliders, 'q' to quit

Keys:
 - Left mouse: click-drag on "Image" to select ROI
 - 'a': Auto thresholds from current ROI (or whole image if no ROI)
 - 's': Save current mask + params JSON
 - 'r': Reset sliders
 - 'q' or ESC: Quit
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

try:
	import cv2
except Exception as e:  # pragma: no cover
	print("OpenCV (cv2) is required. Please install opencv-python.")
	raise


@dataclass
class HSVParams:
	h_min: int = 0
	s_min: int = 0
	v_min: int = 0
	h_max: int = 179
	s_max: int = 255
	v_max: int = 255
	blur: int = 0  # trackbar value, kernel size will be 2*blur+1 if blur>0
	morph: int = 0  # trackbar value, kernel size 2*morph+1, closing iterations = morph


class ROISelector:
	def __init__(self, window: str, image: np.ndarray):
		self.window = window
		self.image = image
		self.drawing = False
		self.ix, self.iy = -1, -1
		self.x, self.y, self.w, self.h = 0, 0, 0, 0
		cv2.setMouseCallback(self.window, self._mouse)

	def _mouse(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix, self.iy = x, y
			self.x, self.y, self.w, self.h = x, y, 0, 0
		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			self.x = min(self.ix, x)
			self.y = min(self.iy, y)
			self.w = abs(x - self.ix)
			self.h = abs(y - self.iy)
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			self.x = min(self.ix, x)
			self.y = min(self.iy, y)
			self.w = abs(x - self.ix)
			self.h = abs(y - self.iy)

	def rect(self) -> Optional[Tuple[int, int, int, int]]:
		if self.w > 2 and self.h > 2:
			return self.x, self.y, self.w, self.h
		return None


def percentile_min_max(values: np.ndarray, p_low=5.0, p_high=95.0) -> Tuple[int, int]:
	low = int(np.percentile(values, p_low))
	high = int(np.percentile(values, p_high))
	return low, high


def auto_threshold_from_roi(hsv: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> HSVParams:
	if roi is not None:
		x, y, w, h = roi
		region = hsv[y : y + h, x : x + w]
		H = region[:, :, 0].reshape(-1)
		S = region[:, :, 1].reshape(-1)
		V = region[:, :, 2].reshape(-1)
	else:
		H = hsv[:, :, 0].reshape(-1)
		S = hsv[:, :, 1].reshape(-1)
		V = hsv[:, :, 2].reshape(-1)

	h_min, h_max = percentile_min_max(H, 2.0, 98.0)
	s_min, s_max = percentile_min_max(S, 5.0, 95.0)
	v_min, v_max = percentile_min_max(V, 5.0, 95.0)

	# Expand a bit to be tolerant
	h_min = max(0, h_min - 2)
	h_max = min(179, h_max + 2)
	s_min = max(0, s_min - 5)
	s_max = min(255, s_max + 5)
	v_min = max(0, v_min - 5)
	v_max = min(255, v_max + 5)

	return HSVParams(h_min, s_min, v_min, h_max, s_max, v_max)


def apply_mask(image_bgr: np.ndarray, hsv: np.ndarray, params: HSVParams) -> Tuple[np.ndarray, np.ndarray]:
	lower = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
	upper = np.array([params.h_max, params.s_max, params.v_max], dtype=np.uint8)
	mask = cv2.inRange(hsv, lower, upper)

	if params.blur > 0:
		k = 2 * params.blur + 1
		mask = cv2.GaussianBlur(mask, (k, k), 0)

	if params.morph > 0:
		mk = 2 * params.morph + 1
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.morph)

	result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
	return mask, result


def ensure_window(name: str):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(name, 640, 480)


def create_trackbars(win: str, params: HSVParams):
	def limit(name, maxv):
		if cv2.getTrackbarPos(name, win) > maxv:
			cv2.setTrackbarPos(name, win, maxv)

	cv2.createTrackbar("H_min", win, params.h_min, 179, lambda v: None)
	cv2.createTrackbar("H_max", win, params.h_max, 179, lambda v: None)
	cv2.createTrackbar("S_min", win, params.s_min, 255, lambda v: None)
	cv2.createTrackbar("S_max", win, params.s_max, 255, lambda v: None)
	cv2.createTrackbar("V_min", win, params.v_min, 255, lambda v: None)
	cv2.createTrackbar("V_max", win, params.v_max, 255, lambda v: None)
	cv2.createTrackbar("Blur", win, params.blur, 10, lambda v: None)
	cv2.createTrackbar("Morph", win, params.morph, 10, lambda v: None)

	# Ensure initial within bounds
	limit("H_min", 179)
	limit("H_max", 179)
	limit("S_min", 255)
	limit("S_max", 255)
	limit("V_min", 255)
	limit("V_max", 255)


def read_trackbars(win: str) -> HSVParams:
	return HSVParams(
		h_min=cv2.getTrackbarPos("H_min", win),
		s_min=cv2.getTrackbarPos("S_min", win),
		v_min=cv2.getTrackbarPos("V_min", win),
		h_max=cv2.getTrackbarPos("H_max", win),
		s_max=cv2.getTrackbarPos("S_max", win),
		v_max=cv2.getTrackbarPos("V_max", win),
		blur=cv2.getTrackbarPos("Blur", win),
		morph=cv2.getTrackbarPos("Morph", win),
	)


def clamp_params(params: HSVParams) -> HSVParams:
	params.h_min = max(0, min(179, params.h_min))
	params.h_max = max(0, min(179, params.h_max))
	params.s_min = max(0, min(255, params.s_min))
	params.s_max = max(0, min(255, params.s_max))
	params.v_min = max(0, min(255, params.v_min))
	params.v_max = max(0, min(255, params.v_max))
	params.blur = max(0, min(10, params.blur))
	params.morph = max(0, min(10, params.morph))
	if params.h_min > params.h_max:
		params.h_min, params.h_max = params.h_max, params.h_min
	if params.s_min > params.s_max:
		params.s_min, params.s_max = params.s_max, params.s_min
	if params.v_min > params.v_max:
		params.v_min, params.v_max = params.v_max, params.v_min
	return params


def save_mask_and_params(image_path: str, mask: np.ndarray, params: HSVParams):
	base, ext = os.path.splitext(image_path)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	mask_path = f"{base}_mask_{ts}.png"
	json_path = f"{base}_hsv_{ts}.json"
	cv2.imwrite(mask_path, mask)
	with open(json_path, "w") as f:
		json.dump(params.__dict__, f, indent=2)
	print(f"Saved: {mask_path}\nSaved: {json_path}")


def draw_roi_rectangle(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
	if roi is None:
		return img
	x, y, w, h = roi
	out = img.copy()
	cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return out


def main():
	parser = argparse.ArgumentParser(description="Interactive HSV Color Finder")
	parser.add_argument("--image", required=True, help="Path to input image")
	args = parser.parse_args()

	image_path = args.image
	if not os.path.isfile(image_path):
		print(f"Image not found: {image_path}")
		sys.exit(1)

	image_bgr = cv2.imread(image_path)
	if image_bgr is None:
		print(f"Failed to load image: {image_path}")
		sys.exit(1)

	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

	ensure_window("Image")
	ensure_window("Mask")
	ensure_window("Result")
	# A small control window to host trackbars
	cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Controls", 480, 320)
	# Create a tiny image as a placeholder for the trackbar host window
	control_canvas = np.zeros((20, 480, 3), dtype=np.uint8)
	cv2.imshow("Controls", control_canvas)

	params = HSVParams()
	create_trackbars("Controls", params)
	roi_selector = ROISelector("Image", image_bgr)

	print("Controls: H/S/V min/max, Blur, Morph. Keys: a(auto), s(save), r(reset), q/ESC(quit)")

	while True:
		params = clamp_params(read_trackbars("Controls"))
		mask, result = apply_mask(image_bgr, hsv, params)

		roi_rect = roi_selector.rect()
		img_with_roi = draw_roi_rectangle(image_bgr, roi_rect)

		cv2.imshow("Image", img_with_roi)
		cv2.imshow("Mask", mask)
		cv2.imshow("Result", result)

		key = cv2.waitKey(30) & 0xFF
		if key in (27, ord('q')):
			break
		elif key == ord('r'):
			params = HSVParams()
			create_trackbars("Controls", params)  # reset positions
		elif key == ord('s'):
			save_mask_and_params(image_path, mask, params)
		elif key == ord('a'):
			auto = auto_threshold_from_roi(hsv, roi_rect)
			auto = clamp_params(auto)
			# Update trackbars to auto values
			cv2.setTrackbarPos("H_min", "Controls", auto.h_min)
			cv2.setTrackbarPos("H_max", "Controls", auto.h_max)
			cv2.setTrackbarPos("S_min", "Controls", auto.s_min)
			cv2.setTrackbarPos("S_max", "Controls", auto.s_max)
			cv2.setTrackbarPos("V_min", "Controls", auto.v_min)
			cv2.setTrackbarPos("V_max", "Controls", auto.v_max)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

