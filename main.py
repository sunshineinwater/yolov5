import torch

yaml_file = "/Users/zhangxuewei/Documents/GitHub/yolov5/data/Objects365.yaml"
# Model
model = torch.hub.load(
    './', 'yolov5s',
    source="local")  # or yolov5n - yolov5x6, custom
# Images
img = '/Users/zhangxuewei/Documents/GitHub/yolov5/data/images/pic.jpeg'  # or file, Path, PIL, OpenCV, numpy, list
# Inference
# model.yaml_file = "/Users/zhangxuewei/Documents/GitHub/yolov5/data/Objects365.yaml"
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
xx1 = results.xyxy[0]  # img1 predictions (tensor)
xx2 = results.pandas().xyxy[0]  # img1 predictions (pandas)

print(xx2)
