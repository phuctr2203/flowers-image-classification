import sys
import os
import torch
from torchvision import transforms
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_model():
    # Get current working directory
    cwd = os.getcwd()
    if torch.cuda.is_available():
        model = torch.load(os.path.join(cwd, 'model.pth'))
    else:
        model = torch.load(os.path.join(cwd, 'model.pth'), map_location=torch.device('cpu'))
    return model

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def classify_image(net, image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = test_transform(image)
    inputs = inputs.to(device)
    inputs = torch.unsqueeze(inputs, 0)
    outputs = net(inputs)
    # print(outputs)
    return torch.argmax(outputs,-1)


model = fetch_model()
image = sys.argv[1]
result = classify_image(model, image)

classes = ['Babi', 'Calimero','Hydrangeas', 'Lisianthus', 'PingPong', "Rosy", "Tana", "Daisy"]

print("This is a", classes[result], "flower.")