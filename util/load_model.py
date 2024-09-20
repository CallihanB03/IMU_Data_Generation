import torch
from classification_models.simple_classification import Classifier

def load_classification_model(input_dim, hidden_dim, output_dim, path):
    classifier = Classifier(input_dim, hidden_dim, output_dim)
    classifier.load_state_dict(torch.load(path, weights_only=True))
    classifier.eval()
    return classifier