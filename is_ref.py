import torch
import torch.nn as nn
from safetensors.torch import load_file
import numpy as np
from sklearn.preprocessing import LabelEncoder
import spacy

# Load SpaCy model for vectorization
nlp = spacy.load("en_core_web_lg") #replace with en_core_web_sm for a lighter load


# Define the model architecture (must match the original architecture)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Load the saved model and tensors
class ReferenceClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the classifier by loading the model and setting up the label encoder.
        :param model_path: Path to the trained model saved in safetensors format.
        """
        self.input_dim = 300  # SpaCy vectors are 300-dimensional
        self.output_dim = 2   # Two classes: 'ref' and 'not-ref'
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['not-ref', 'ref'])  # Predefined classes

        # Load model
        self.model = SimpleNN(self.input_dim, self.output_dim)
        loaded_tensors = load_file(model_path)
        self.model.load_state_dict(loaded_tensors)
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        Predict the class for a single string.
        :param text: Input text string to classify.
        :return: Predicted class ('ref' or 'not-ref').
        """
        vector = torch.tensor(nlp(text).vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(vector)
            _, predicted_label = torch.max(output, 1)
            predicted_class = self.label_encoder.inverse_transform(predicted_label.cpu().numpy())
        return predicted_class[0]

    def predict_batch(self, text_array: np.ndarray) -> np.ndarray:
        """
        Predict the class for a numpy array of strings.
        :param text_array: Numpy array of strings to classify.
        :return: Numpy array of predicted classes ('ref' or 'not-ref').
        """
        predictions = []
        for text in text_array:
            predictions.append(self.predict(str(text)))
        return np.array(predictions)
