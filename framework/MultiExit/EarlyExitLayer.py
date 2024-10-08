from torch import nn

class EarlyExitLayer(nn.Module):
    def __init__(self, layers, num_classes, exit_layers=None, pool=True, dropout=True):
        super(EarlyExitLayer, self).__init__()
        self.layers = nn.Sequential(*layers) # feature extractor
        self.exit_layers = exit_layers
        if self.exit_layers is None:
            self.exit_layers = nn.Sequential()

        self.flatten_layers = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
        ) if pool else nn.Flatten()
        
        self.add_dropout    = dropout
        self.is_initialized = False
        self.num_classes    = num_classes
        
    def forward(self, x):
        if self.is_initialized:
            feature = self.layers(x)
            flatten_feature = self.flatten_layers(feature)
            preds = self.exit_layers(flatten_feature)

            return feature, flatten_feature, preds

        flatten_size = self.flatten_layers(self.layers(x)).shape[1]

        if self.add_dropout:
            self.exit_layers.add_module('dropout', nn.Dropout(0.5))
        self.exit_layers.add_module('fc', nn.Linear(flatten_size, self.num_classes))
        self.is_initialized = True

        return self.forward(x)