import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm

# Define uma classe de conjunto de dados personalizada
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img_name) for img_name in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)

        if self.transform:
            # Converte o tensor de imagem de volta para uma imagem PIL
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        
        # Extrai a classe do nome da imagem (com base no primeiro caractere)
        class_name = os.path.basename(img_path)[0]
        
        return image, class_name  # Retorna o primeiro caractere do nome da classe

# Define transformações de dados (mesmas transformações do código anterior)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Carrega o modelo treinado
model = models.resnet50(pretrained=False)
num_classes = 4  # Substitua pelo número correto de classes do seu modelo
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('C:\\Users\\felip\\Documents\\IC_Projeto\\ClassificationModel\\resnet50_4000_images.pth'))
model.eval()

# Cria um conjunto de dados para imagens não vistas
test_dataset = CustomImageDataset(root_dir='C:\\Users\\felip\\Documents\\IC_Projeto\\dataset\\RealImages', transform = data_transform)

# Cria um DataLoader para as novas imagens
test_dataloader = DataLoader(test_dataset, batch_size =  64, shuffle=False)

# Loop para fazer previsões nas imagens não vistas
correct = 0
total = 0

print("Results for Unseen Images:")
for images, class_names in tqdm(test_dataloader, desc = 'Testing', dynamic_ncols=True):
    with torch.no_grad():
        images = images.float()  # Converter a imagem para float
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(class_names)
        
        # Imprime alguns dos resultados
        for i in range(len(class_names)):
            image_name = os.path.basename(class_names[i])  # Nome da imagem
            predicted_class = predicted[i].item()  # Classe prevista
            print(f"Image: {image_name}, Predicted Class: {predicted_class}")
            
            # Verifica se a previsão está correta
            if predicted_class == int(image_name[0]):
                correct += 1

# Calcula a taxa de acerto
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
