from DataLoader import LeafDiseaseDataset, train_val_dataset
from torchvision.models import resnet18
from torch import nn
import torch
import numpy as np
from torch import optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

class Model():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=True).to(self.device)


    def accuracy(self, out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()

    def train(self, train_dataloader, test_dataloader):
        n_epochs = 5
        NUM_CLASS = 5
        use_cuda = torch.cuda.is_available()
        print_every = 10
        valid_loss_min = np.Inf
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        total_step = len(train_dataloader)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, NUM_CLASS).to(self.device)
        #self.model.fc = self.model.fc.cuda() if use_cuda else self.model.fc
        summary(self.model, (3, 224, 224))

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_,target_) in enumerate(train_dataloader):
                #print(data_.shape, data_)
                #data_ = np.transpose(data_, (0,3,1,2))
                #data_ = torch.from_numpy(data_)
                #target_ = torch.from_numpy(np.array([target_]))
                data_, target_ = data_.to(self.device, dtype=torch.float), target_.to(self.device, dtype=torch.long)
                optimizer.zero_grad()

                outputs = self.model(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred == target_).item()
                total += target_.size(0)
                if (batch_idx) % 20 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss / total_step)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
            batch_loss = 0
            total_t = 0
            correct_t = 0
            with torch.no_grad():
                self.model.eval()
                for data_t, target_t in (test_dataloader):
                    data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                    outputs_t = self.model(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t == target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t / total_t)
                val_loss.append(batch_loss / len(test_dataloader))
                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(self.model.state_dict(), 'resnet.pt')
                    print('Improvement-Detected, save-model')

    def test(self):
        pass


def main():
    model = Model()
    CSV_PATH = 'Data/train.csv'
    ROOT_PATH = 'Data/train_images'
    dataset = LeafDiseaseDataset(csv_file=CSV_PATH, root_dir=ROOT_PATH)
    train_dataset, test_dataset = train_val_dataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    model.train(train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()

