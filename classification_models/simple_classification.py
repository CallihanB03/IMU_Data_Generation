import torch
import torch.nn as nn 
import numpy as np 
from util.participant_data import load_participant_data
from util.pre_processing import df_to_tensor, tensor_train_test_split
from util.label_encoding import dummy_to_one_hot

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.num_classes)

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.train_losses = []
        self.test_losses = []

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        return x
    
    def epoch_train(self, train_data, train_labels, test_data, test_labels, epochs, loss, optimizer, test_freq=5):
        self.train_losses = np.zeros(epochs)
        self.test_losses = np.zeros(epochs // test_freq)
        test_loss_ind = 0

        for epoch in range(epochs):
            pred_labels = self(train_data)

            epoch_loss = 5 * loss(pred_labels, train_labels)
            self.train_losses[epoch] = epoch_loss
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            display_loss = "{:.4}".format(epoch_loss)

            print(f"Epoch: {epoch+1}, Loss: {display_loss}")

            if not epoch % test_freq:
                epoch_test_loss = self.evaluate(test_data, test_labels, loss=loss)
                self.test_losses[test_loss_ind] = epoch_test_loss
                display_test_loss = "{:.4}".format(epoch_test_loss)
                print(f"Test Loss: {display_test_loss}")
                test_loss_ind += 1


    def epsilon_train(self, train_data, train_labels, test_data, test_labels, loss, optimizer, epsilon=0.01, test_freq=5):
        prev_loss = None
        self.train_losses = []
        self.test_losses = []
        epoch = 0

        while True:
            pred_labels = self(train_data)

            epoch_loss = 5 * loss(pred_labels, train_labels)
            self.train_losses.append(epoch_loss)
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            display_loss = "{:.4}".format(epoch_loss)

            print(f"Epoch: {epoch+1}, Loss: {display_loss}")

            if not epoch % test_freq:
                epoch_test_loss = self.evaluate(test_data, test_labels, loss=loss)
                self.test_losses.append(epoch_test_loss)
                display_test_loss = "{:.4}".format(epoch_test_loss)
                print(f"Test Loss: {display_test_loss}")

            if prev_loss and round(abs(prev_loss - epoch_loss).item(), 3) <= epsilon:
                break

            prev_loss = epoch_loss




    def evaluate(self, data, label, loss):
        pred_y = self(data)
        test_loss = loss(pred_y, label)
        return test_loss


if __name__ == "__main__":
    data = load_participant_data()

    data = data[data.columns[:-2]] # Removing string arguments
    X, y = df_to_tensor(data[data.columns[:-1]]), df_to_tensor(data[data.columns[-1]])
    y = dummy_to_one_hot(y)


    x_train, y_train, x_test, y_test = tensor_train_test_split(X, y, test_size=0.2)

    input_dim = x_train.shape[1]
    num_classes = 3

    loss = nn.MSELoss()



    classifier = Classifier(input_dim=input_dim, 
                            hidden_dim=24, 
                            num_classes=num_classes)

    opt = torch.optim.Adam(params=classifier.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    classifier.epoch_train(x_train, y_train, x_test, y_test, epochs=10, loss=loss, optimizer=opt)
    #classifier.epsilon_train(x_train, y_train, x_test, y_test, loss=loss, optimizer=opt)

    print(f"train losses = {classifier.train_losses}")
    print(f"test losses = {classifier.test_losses}")

