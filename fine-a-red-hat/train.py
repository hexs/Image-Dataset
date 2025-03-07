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

if __name__ == '__main__':
    model = YOLO('yolo11m.pt')  # Create a new model
    # model = YOLO(r'runs\detect\train\weights\last.pt')  # Continue training

    results = model.train(data='data.yaml', epochs=5)

    results = model.val()  # Test the model

    success = model.export(format='onnx')  # Save the model in ONNX format.

    print(results)
    print(success)
