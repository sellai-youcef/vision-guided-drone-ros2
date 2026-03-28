from ultralytics import YOLO

model = YOLO('yolo11m.pt')

results = model.train(

data='data.yaml',
epochs=50,
imgsz=640,
batch=32,
name='drone_detector',
device=0

)

print("Trainind done")
print(f"Best model saved at: {results.save_dir}")