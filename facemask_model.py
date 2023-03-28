import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from facemask_reg import *

class FacemaskRecognizeModel(nn.Sequential):
    def __init__(self):
        super(FacemaskRecognizeModel, self).__init__(
            # first layer
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # second layer
            nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # flatten
            nn.Flatten(),

            # linear layers
            nn.Linear(in_features = 300, out_features = 150),
            nn.Linear(in_features = 150, out_features = 1),

            # active function
            nn.Sigmoid()
        )



def save_model(model, path):
    torch.save(model.state_dict(), path)



def load_pretrain(model, path):
    model.load_state_dict(torch.load(path))



def predict(model, image):
    transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])
    
    data = transform(image)
    data = data.unsqueeze(0)
    
    prob = model(data)
    ans = torch.round(prob).int()
    return ans



def create_dataset(path):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    return dataset



def create_dataloaders(dataset, train_ratio, test_ratio):
    train_dataset, test_dataset = random_split(dataset, [train_ratio, test_ratio])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=2)
    return train_dataloader, test_dataloader



def train(model, train_dataloader, n_epoch, learning_rate):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    for epoch in range(n_epoch):
        
        for batch, (images, labels) in enumerate(train_dataloader):
            predicts = model(images)
            
            labels = labels.view(-1, 1).float()
            loss = loss_fn(predicts, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 1000 == 0:
                print('epoch {}\t\t batch {}\t\t loss {}'.format(epoch, batch, loss))



def evaluate(model, test_dataloader):
    n_predict = 0
    n_correct = 0

    for batch, (images, labels) in enumerate(test_dataloader):
        predicts = model(images)
        predicts = torch.round(predicts).int()
    
        labels = labels.view(-1, 1)
        equals = torch.eq(predicts, labels)
        
        n_predict += len(predicts)
        n_correct += equals.sum()

    return n_predict, n_correct



def main():
    dataset = create_dataset(path = 'data_extracted/')
    train_dataloader, test_dataloader = create_dataloaders(dataset, train_ratio = 0.8, test_ratio = 0.2)
    model = FacemaskRecognizeModel()
    train(model, train_dataloader, n_epoch = 2, learning_rate = 0.01)

    n_predict, n_correct = evaluate(model, test_dataloader)
    print('n_predict', n_predict)
    print('n_correct', n_correct)



if __name__ == "__main__":
    main()