from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from cnn import SimpleCNN
app = FastAPI()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
model = SimpleCNN()
model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return {"message": "CNN CIFAR10 Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
    predicted_class = classes[prediction.item()]

    return {
        "prediction": predicted_class
    }
