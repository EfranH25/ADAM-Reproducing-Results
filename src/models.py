from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class MultiNet(nn.Module):
    def __init__(self, input_dim, num_class, dropout_prob=0.5):
        super(MultiNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1000, num_class),
        )

    def forward(self, x):
        return self.classifier(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
