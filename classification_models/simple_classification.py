import os
import torch
import torch.nn as nn 
import numpy as np 
from statistics import mean
from util.participant_data import load_participant_data
from util.pre_processing import df_to_tensor, tensor_train_test_split
from util.label_encoding import dummy_to_one_hot, one_hot_to_dummy_encoding
from util.model_metrics import precision, recall, f1


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.optimal_params = self.state_dict()
        self.lowest_test_loss = None


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
        self.train_losses = []
        self.test_losses = []
        test_loss_ind = 0

        for epoch in range(epochs):
            pred_labels = self(train_data)

            epoch_loss = 8 * loss(pred_labels, train_labels)
            self.train_losses.append(epoch_loss)
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            display_loss = "{:.4}".format(epoch_loss)

            print(f"Epoch: {epoch+1}, Loss: {display_loss}")

            if not epoch % test_freq:
                epoch_test_loss = self.evaluate(test_data, test_labels, loss=loss)

                # saving optimal model params
                if self.lowest_test_loss and epoch_test_loss < self.lowest_test_loss:
                    self.lowest_test_loss = epoch_test_loss
                    self.optimal_params = self.state_dict()

                self.test_losses.append(epoch_test_loss)
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

            epoch_loss = 8 * loss(pred_labels, train_labels)
            self.train_losses.append(epoch_loss)
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            display_loss = "{:.4}".format(epoch_loss)

            print(f"Epoch: {epoch+1}, Loss: {display_loss}")

            if not epoch % test_freq:
                epoch_test_loss = self.evaluate(test_data, test_labels, loss=loss)

                # saving optimal model params
                if self.lowest_test_loss and epoch_test_loss < self.lowest_test_loss:
                    self.lowest_test_loss = epoch_test_loss
                    self.optimal_params = self.state_dict()

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
    
    def save_model(self, save_title="saved_model"):
        path = "./saved_models/"
        save_path = path + save_title + ".pth"

        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")

        torch.save(self.optimal_params, save_path)
        print(f"model saved to {save_path}")
        return None
    
    def evaluate_metrics(self, input_data, labels):
        num_metrics = 3
        num_classes = labels.shape[1]
        metrics = torch.zeros(num_classes, num_metrics)

        for label in range(num_classes):
            prec = precision(self, input_data, labels, precison_wrt=label)
            rec = recall(self, input_data, labels, recall_wrt=label)
            f_1 = f1(self, input_data, labels, f1_wrt=label)

            metrics[label][0] = prec
            metrics[label][1] = rec
            metrics[label][2] = f_1

        return metrics
    
    def show_metrics(self, input_data, labels):
        metrics = self.evaluate_metrics(input_data, labels)
        print(metrics) 
        return None
        

    
    def confusion_matrix(self, data, label, normalize=True):
        """
        cols iterate through y_pred
        rows iterate through y_true
        """
        y_pred = self(data)
        num_classes = label.shape[1]

        y_true, y_pred = one_hot_to_dummy_encoding(label), one_hot_to_dummy_encoding(y_pred)


        assert y_pred.shape == y_true.shape, f"shape of predicted labels {y_pred.shape} is incompatible with shape of true labels {y_true.shape}" 

        conf_matr = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                count = 0

                for k, _ in enumerate(y_pred):
                    if y_pred[k] == i and y_true[k] == j:
                        count += 1
                
                conf_matr[i][j] = count

        if normalize:
            total = y_pred.shape[0]
            for i in range(num_classes):
                for j in range(num_classes):
                    conf_matr[i][j] = round((conf_matr[i][j] / total).item(), 2)

        return conf_matr



if __name__ == "__main__":
    data = load_participant_data()

    data = data[data.columns[:-2]] # Removing string arguments
    X, y = df_to_tensor(data[data.columns[:-1]]), df_to_tensor(data[data.columns[-1]])
    y = dummy_to_one_hot(y)


    x_train, y_train, x_test, y_test = tensor_train_test_split(X, y, test_size=0.2)

    input_dim = x_train.shape[1]
    num_classes = 3



    classifier = Classifier(input_dim=input_dim, 
                            hidden_dim=24, 
                            num_classes=num_classes)

    opt = torch.optim.Adam(params=classifier.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()


    epochs = 10000

    classifier.epoch_train(x_train, y_train, x_test, y_test, epochs=epochs, loss=loss, optimizer=opt)
    metrics = classifier.evaluate_metrics(x_test, y_test)
    f1_scores = [metrics[i][2].item() for i in range(num_classes)]

    avg_f1 = round(mean(f1_scores), 3)
    print(f"average f1 score = {avg_f1}")

    classifier.save_model(save_title=f"epochs_{epochs}_average_f1_{avg_f1}")
    

    

