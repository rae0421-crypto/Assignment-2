import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import get_model
from trainer import train_model
from evaluator import evaluate_model
EPOCHS = 10
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = get_model("CNN").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, optimizer, criterion, train_loader, device, epochs=EPOCHS)
accuracy = evaluate_model(model, test_loader, device)
torch.save(model.state_dict(), "cnn_model.pth")
print(f"Model saved. Final Accuracy: {accuracy:.2f}%")
