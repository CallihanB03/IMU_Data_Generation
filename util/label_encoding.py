import torch

def one_hot_to_dummy_encoding(labels):
    num_data_points, _ = labels.shape
    dummy_encodings = torch.zeros(num_data_points)

    for row in range(num_data_points):
        label = float(torch.argmax(labels[row]))
        dummy_encodings[row] = label
    
    return dummy_encodings


if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import df_to_tensor
    from classification_models.simple_classification import Classifier
    import torch.nn as nn

    data = load_participant_data()
    data = data[data.columns[:-2]] # Removing Columns
    data = df_to_tensor(data)

    input_dim = data.shape[1]
    num_classes = 3

    classifier = Classifier(input_dim=input_dim, 
                            hidden_dim=24, 
                            num_classes=num_classes)
    
    opt = torch.optim.Adam(params=classifier.parameters(), lr=1e-3)
    loss = nn.MSELoss()



    results = classifier(data)

    breakpoint()

