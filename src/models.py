import torch
from babel.messages.extract import extract
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

class ConvNetDrop(nn.Module):
    def __init__(self, dropout_chance=0.5):
        super(ConvNetDrop, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_chance),
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
            nn.Dropout(dropout_chance),
            nn.Linear(in_features=1000, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Vision Transformer model from following tutorial
# https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch4_Vision_Transformers.ipynb

def extract_patches(image_tensor, patch_size=8):
    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()
    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)
    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    return unfolded

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mutlihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Multi-Layer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ELU(),
            nn.Linear(hidden_size*2, hidden_size)
        )

    def forward(self, x):
        # Apply the first layer normalization
        norm_x = self.norm1(x)
        # Apply multi-head attention and add the input (residual connection)
        x = self.mutlihead_attn(norm_x, norm_x, norm_x)[0] + x
        # Apply the second layer normalization
        norm_x = self.norm2(x)
        # Pass through the MLP and add the input (residual connection)
        x = self.mlp(norm_x) + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, channels_in, patch_size, hidden_size, num_layers, num_heads=8):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size

        # Fully connected layer to project input patches to the hidden size dimension
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        # Create a list of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        # Fully connected output layer to map to the number of classes (e.g., 10 for CIFAR-10)
        self.fc_out = nn.Linear(hidden_size, 10)
        # Parameter for the output token
        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Positional embeddings to retain positional information of patches
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.001))


    def forward(self, image):
            bs = image.shape[0]
            # Extract patches from the image and flatten them
            patch_seq = extract_patches(image, patch_size=self.patch_size)
            # Project patches to the hidden size dimension
            patch_emb = self.fc_in(patch_seq)
            # Add positional embeddings to the patch embeddings
            patch_emb = patch_emb + self.pos_embedding
            # Concatenate the output token to the patch embeddings
            embs = torch.cat((self.out_vec.expand(bs, 1, -1), patch_emb), 1)
            # Pass the embeddings through each Transformer block
            for block in self.blocks:
                embs = block(embs)
            # Use the embedding of the output token for classification
            return self.fc_out(embs[:, 0])
