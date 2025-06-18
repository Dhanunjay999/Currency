# # from flask import Flask, request, jsonify
# # import torch
# # from PIL import Image
# # import io
# # import pathlib
# # import numpy as np

# # pathlib.PosixPath = pathlib.WindowsPath
# # app = Flask(__name__)

# # # Load YOLOv5 model
# # model = torch.hub.load('.', 'custom', path='best.pt', source='local')
# # model.eval()
# # model.conf = 0.1  # Lower threshold

# # @app.route('/detect', methods=['POST'])
# # def detect():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image provided'}), 400

# #     file = request.files['image']
# #     try:
# #         img_bytes = file.read()
# #         img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
# #         img_np = np.array(img)
# #     except Exception as e:
# #         return jsonify({'error': f'Invalid image: {str(e)}'}), 400

# #     results = model(img_np)
# #     predictions = results.pandas().xyxy[0]
# #     print(predictions)  # Debug

# #     if predictions.empty:
# #         return jsonify({'message': 'No objects detected'}), 200

# #     detected = {}
# #     for _, row in predictions.iterrows():
# #         label = row['name']
# #         detected[label] = detected.get(label, 0) + 1

# #     return jsonify({'detections': detected})

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=8080)


# from flask import Flask, request, jsonify
# import torch
# import numpy as np
# from PIL import Image
# import sys
# from pathlib import Path
# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath

# # Set root directory
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT))

# from models.yolo import Model
# from utils.augmentations import letterbox
# from utils.general import non_max_suppression
# from utils.torch_utils import select_device

# app = Flask(__name__)

# # Load model
# weights_path = "C:\\Users\\kopparapu\\Desktop\\IndianCurrencyNotesDetection\\best_10_20.pt"
# device = select_device('cpu')  # or 'cuda:0' if GPU

# # Load checkpoint
# ckpt = torch.load(weights_path, map_location=device)
# model = Model(ckpt['model'].yaml).to(device)
# model.load_state_dict(ckpt['model'].state_dict())
# model.eval()

# conf_thres = 0.25
# iou_thres = 0.45
# stride = int(model.stride.max())

# # Define class names
# names = ['0', '1', '2', '3', '4', '5', '6', '7']

# # Manually add scale_coords
# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     if ratio_pad is None:  # Calculate from img0_shape
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#         pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
#                (img1_shape[0] - img0_shape[0] * gain) / 2)  # width, height padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     coords[:, :4] = coords[:, :4].clamp(min=0)
#     return coords

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     try:
#         file = request.files['image']
#         img0 = Image.open(file.stream).convert('RGB')
#         img0 = np.array(img0)
#     except Exception as e:
#         return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

#     # Preprocess
#     img = letterbox(img0, new_shape=640, stride=stride)[0]
#     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device).float() / 255.0
#     img = img.unsqueeze(0)

#     # Inference
#     with torch.no_grad():
#         pred = model(img)[0]
#         pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

#     best_result = None
#     best_conf = -1
#     if pred[0] is not None and len(pred[0]):
#         pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred[0]:
#             conf_val = float(conf)
#             if conf_val > best_conf:
#                 best_conf = conf_val
#                 best_result = {
#                     'class': names[int(cls)],
#                     'confidence': round(conf_val, 4),
#                     'bbox': [round(float(x), 2) for x in xyxy]
#                 }

#     if not best_result:
#         return jsonify({'message': 'No currency notes detected.'}), 200

#     return jsonify({'prediction': best_result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)
# from flask import Flask, request, jsonify
# import torch
# import numpy as np
# from PIL import Image
# import sys
# from pathlib import Path
# import pathlib

# # Patch for Windows Path compatibility
# pathlib.PosixPath = pathlib.WindowsPath

# # Set root directory and ensure access to YOLOv5 internals
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT / "yolov5"))

# from models.yolo import Model
# from utils.augmentations import letterbox
# from utils.general import non_max_suppression
# from utils.torch_utils import select_device

# app = Flask(__name__)

# # Load model
# weights_path = "C:\\Users\\kopparapu\\Desktop\\IndianCurrencyNotesDetection\\yolov5\\runs\\train\\currency_exp4\\weights\\best.pt"  # Update if your model has a different name
# device = select_device('cpu')  # Use 'cuda:0' if you want GPU

# # Load checkpoint
# ckpt = torch.load(weights_path, map_location=device)
# model = Model(ckpt['model'].yaml).to(device)
# model.load_state_dict(ckpt['model'].state_dict())
# model.eval()

# # Get class names from model
# if hasattr(ckpt['model'], 'names'):
#     names = ckpt['model'].names
# else:
#     names = [str(i) for i in range(ckpt['model'].nc)]

# conf_thres = 0.4
# iou_thres = 0.45
# stride = int(model.stride.max())

# # Scale coordinates back to original image
# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     if ratio_pad is None:
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
#         pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
#                (img1_shape[0] - img0_shape[0] * gain) / 2)
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     coords[:, :4] = coords[:, :4].clamp(min=0)
#     return coords

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     try:
#         file = request.files['image']
#         img0 = Image.open(file.stream).convert('RGB')
#         img0 = np.array(img0)
#     except Exception as e:
#         return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

#     # Preprocess image
#     img = letterbox(img0, new_shape=640, stride=stride)[0]
#     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device).float() / 255.0
#     img = img.unsqueeze(0)

#     # Inference
#     with torch.no_grad():
#         pred = model(img)[0]
#         pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

#     best_result = None
#     best_conf = -1
#     if pred[0] is not None and len(pred[0]):
#         pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred[0]:
#             conf_val = float(conf)
#             if conf_val > best_conf:
#                 best_conf = conf_val
#                 best_result = {
#                     'class': names[int(cls)],
#                     'confidence': round(conf_val, 4),
#                     'bbox': [round(float(x), 2) for x in xyxy]
#                 }

#     if not best_result:
#         return jsonify({'message': 'No currency notes detected.'}), 200

#     return jsonify({'prediction': best_result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)


import torch
import cv2
import matplotlib.pyplot as plt

# Load model
model = torch.hub.load('.', 'custom', path='C:\\Users\\kopparapu\\Desktop\\IndianCurrencyNotesDetection\\best (2).pt', source='local')

# Set model confidence threshold if needed (default is 0.25)
model.conf = 0.25

# Path to test image
image_path = 'C:\\Users\\kopparapu\\Desktop\\only 20rs\\dataset_split\\train\\images\\1_jpg.rf.9e8f85c33d23b309edc0077aed84a57f.jpg'

# Load image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inference
results = model(img_rgb)

# Print results
results.print()

# Show results
results.show()

# (Optional) Save the output image
# results.save(save_dir='C:/Users/kopparapu/Desktop/output/')
