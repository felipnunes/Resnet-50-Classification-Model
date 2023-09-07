import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

# Define a custom dataset class
class CustomImageDataset():
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    sample_path = os.path.join(class_dir, filename)
                    class_idx = self.class_to_idx[class_name]
                    samples.append((sample_path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = read_image(img_path)
        
        # Converter para PIL Image
        image_pil = transforms.ToPILImage()(image)
        
        if self.transform:
            image_pil = self.transform(image_pil)
        if self.target_transform:
            label = self.target_transform(label)
        return image_pil, label

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the custom dataset
dataset = CustomImageDataset(root_dir='C:\\Users\\felip\\Documents\\IC_Projeto\\dataset\\Sketch', transform=data_transform)

# Split dataset into train and test sets
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders for batching and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create a ResNet-50 model
model = models.resnet50(pretrained=True)
# Change the output layer to match the number of classes in your dataset
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop (com barra de progresso)
num_epochs = 10
for epoch in range(num_epochs):
    # Crie uma barra de progresso para o dataloader
    dataloader_with_progress = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch [{epoch + 1}/{num_epochs}]', dynamic_ncols=True)
    total_loss = 0  # Inicialize a perda total

    for images, labels in dataloader_with_progress:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # Adiciona a perda atual à perda total

    average_loss = total_loss / len(train_dataloader)  # Calcula a perda média

    # Atualiza a descrição da barra de progresso com a perda média
    dataloader_with_progress.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    # Imprime o valor da perda após cada época
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Test loop
model.eval()
correct = 0
total = 0
for images, labels in tqdm(test_dataloader, desc='Testing', dynamic_ncols=True):
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Save the trained model if needed
torch.save(model.state_dict(), 'resnet50_custom_dataset.pth')
