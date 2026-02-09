import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_utils import get_device
from models import MLP, CNN
from datasets import get_mnist

train_ds, test_ds = get_mnist()
device = get_device()

input_size = 28*28
hidden_size = 512
output_size = 10
learning_rate = 0.001
epochs = 50
batch_size = 156

# Crear los DataLoaders para entrenamiento y prueba
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

#model = MLP(input_size, hidden_size, output_size)
model = CNN(output_size).to(device)
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history  = {'train_loss':[], 'val_loss':[]}
#Entrenamiento
for epoch in range(epochs):
  model.train()
  epoch_loss = 0.0
  for batch_X, batch_y in train_loader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device).long()
    optimizer.zero_grad()
    outputs = model(batch_X) #forward
    loss = loss_function(outputs, batch_y) #perdida
    loss.backward() # Backward // Calcular gradientes
    optimizer.step() #Actualizar parametros
    epoch_loss += loss.item()
  train_loss = epoch_loss / len(train_loader)
  history['train_loss'].append(train_loss)

  #Evaluacion
  model.eval()
  with torch.no_grad():
    epoch_loss = 0.0
    for batch_X, batch_y in test_loader:
      batch_X = batch_X.to(device)
      batch_y = batch_y.to(device).long()
      val_outputs = model(batch_X)
      val_loss = loss_function(val_outputs, batch_y).item()
      epoch_loss += val_loss
    val_loss = val_loss / len(test_loader)
    history['val_loss'].append(val_loss)
  if epoch % 10 == 0:
    print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

torch.save(model.state_dict(), "models/mnist_mlp.pth")
print("Modelo guardado correctamente.")


plt.figure()
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("graphics/loss_plot.png")
plt.show()


all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("graphics/confusion_matrix.png")
plt.show()
print("Matriz de confusión guardada.")


data_iter = iter(test_loader)
images, labels = next(data_iter)
output = model(images)
_, predicted = torch.max(output, 1)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
  axes[i].imshow(images[i].squeeze(), cmap='gray')
  axes[i].set_title(f"Pred: {predicted[i].item()}")
  axes[i].axis('off')
plt.show()
