from roboflow import Roboflow
import os

rf = Roboflow(api_key="YxHugN0GZSrWciAmo8a6")
project = rf.workspace("karol-chmco").project("pcbb-k8dth")
dataset = project.version(3).download("yolov8")

# Update paths in data.yaml
data_yaml = "data_from_dir.yaml"
with open(data_yaml, "r") as f:
    lines = f.readlines()

# Get current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Update paths
for i, line in enumerate(lines):
    if line.startswith("train:"):
        lines[i] = f"train: {os.path.join(current_dir, 'train', 'images')}\n"
    elif line.startswith("val:"):
        lines[i] = f"val: {os.path.join(current_dir, 'valid', 'images')}\n"
    elif line.startswith("test:"):
        lines[i] = f"test: {os.path.join(current_dir, 'test', 'images')}\n"

# Write updated yaml file
with open(data_yaml, "w") as f:
    f.writelines(lines)