import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        # embeddings = torch.cat((embeddings, conv_embeds), 1)

        return self.classifier(embeddings)
