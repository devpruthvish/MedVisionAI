import torch
import torch.nn as nn
from torchvision.models import resnet18


def load_model(model_path):

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model