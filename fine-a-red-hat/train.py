from hexss import check_packages, json_update, username

check_packages(
    'ultralytics', 'onnx', 'opencv-python',
    auto_install=True, verbose=False,
)
from ultralytics import YOLO

from hexss.path import get_script_directory

json_update(rf"C:\Users\{username}\AppData\Roaming\Ultralytics\settings.json", {
    "datasets_dir": rf"{get_script_directory()}\datasets",
    "weights_dir": rf"{get_script_directory()}\weights",
    "runs_dir": rf"{get_script_directory()}\runs",
})

'https://github.com/ultralytics/ultralytics?tab=readme-ov-file#models'

model = YOLO('yolov8m.yaml')  # เป็นการสร้างโมเดลใหม่ขึ้นมา

# โหลด pretrained model มาเพื่อให้เราไม่ต้องเทรนใหม่ทั้งหมดตั้งแต่เริ่ม
# model = YOLO(r'runs\detect\train\weights\last.pt')

results = model.train(data='data.yaml', epochs=1)

# ทดสอบโมเดลโดยใช้ validation datasets ที่เตรียมไว้
results = model.val()

# เซฟโมเดลโดยให้โมเดลอยู่ใน ONNX format
success = model.export(format='onnx')

print(results)
print(success)
