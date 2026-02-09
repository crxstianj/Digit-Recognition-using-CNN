from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from models import MLP, CNN
import os
import io
from fastapi.middleware.cors import CORSMiddleware


input_size = 28*28
#model = MLP(input_size, 128, 10)
model = CNN(10)
model_path = os.path.join('models', "mnist_mlp.pth")
model.load_state_dict(torch.load(model_path,
                                 weights_only=True))
model.eval()

transform = transforms.Compose([
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
  image_bytes = await file.read()
  image = Image.open(io.BytesIO(image_bytes))
  image = image.resize((28, 28))
  image_tensor = transform(image).unsqueeze(0)
  with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    prediction = predicted.item()
    print(prediction)
    return {"prediction": prediction}

if __name__ == '__main__':
  import uvicorn
  uvicorn.run(app, host='127.0.0.1', port=3000)