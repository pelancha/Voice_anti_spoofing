from torch import nn
import torch


class MFM2_1(nn.Module):
    def __init__(self, input_channels, output_size, kernel_size=3, stride=1, padding=1, in_fc_layer=False):
        super().__init__()

        if (output_size != input_channels // 2):
            raise ValueError(f"Output size of MFM layer was expected to be {input_channels // 2}, got {output_size}")
        if not in_fc_layer:
            self.net = nn.Conv2d(input_channels, output_size * 2, kernel_size, stride, padding)
        else:
            self.net = nn.Linear(input_channels, output_size * 2)

    def forward(self, x):
        x = self.net(x)
        out = torch.split(x, x.size(1) // 2, dim = 1)
        return torch.max(out[0], out[1])


class Layer1(nn.Module):
    def __init__(self, input_channels, out_hidden):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, out_hidden * 2, kernel_size=(5, 5), stride=(1, 1), padding=2),
            MFM2_1(out_hidden * 2, out_hidden)
        )

    def forward(self, input_data):
        return self.layer(input_data)


class Layer2(nn.Module):
    def __init__(self, in_hidden, out_hidden):
        super().__init__()
        self.layer = nn.Sequential(       
            nn.Conv2d(in_hidden, in_hidden * 2, kernel_size=(1, 1), stride=(1, 1)),
            MFM2_1(in_hidden * 2, in_hidden),
            nn.BatchNorm2d(in_hidden), 
            nn.Conv2d(in_hidden, out_hidden * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM2_1(out_hidden * 2, out_hidden)
        )

    def forward(self, input_data):
        return self.layer(input_data)


class Layer3(nn.Module):
    def __init__(self, in_hidden, out_hidden):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_hidden, in_hidden * 2, kernel_size=(1, 1), stride=(1, 1)),
            MFM2_1(in_hidden * 2, in_hidden),
            nn.BatchNorm2d(in_hidden),
            nn.Conv2d(in_hidden, out_hidden * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM2_1(out_hidden * 2, out_hidden)
        )

    def forward(self, input_data):
        return self.layer(input_data)


class Layer4(nn.Module):
    def __init__(self, in_hidden, out_hidden):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_hidden, in_hidden * 2, kernel_size=(1, 1), stride=(1, 1)),
            MFM2_1(in_hidden * 2, in_hidden),
            nn.BatchNorm2d(in_hidden),
            nn.Conv2d(in_hidden, in_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM2_1(in_hidden, out_hidden),
            nn.BatchNorm2d(out_hidden),
            nn.Conv2d(out_hidden, in_hidden, kernel_size=(1, 1), stride=(1, 1)),
            MFM2_1(in_hidden, out_hidden),
            nn.BatchNorm2d(out_hidden),
            nn.Conv2d(out_hidden, in_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM2_1(in_hidden, out_hidden)
        )

    def forward(self, input_data):
        return self.layer(input_data)
    

class LCNN(nn.Module):
    def __init__(self, n_feats, fc_hidden, linear_input_size, n_class):
        super().__init__()
        fc_hidden_2 = fc_hidden * 3 // 2    
        
        self.net = nn.Sequential(
            Layer1(n_feats, fc_hidden),
            
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            Layer2(fc_hidden, fc_hidden_2),
            
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(fc_hidden_2),

            Layer3(fc_hidden_2, fc_hidden * 2),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            Layer4(fc_hidden * 2, fc_hidden),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Flatten(),
            nn.Linear(linear_input_size[0] * linear_input_size[1] * fc_hidden, fc_hidden * 5),
            MFM2_1(fc_hidden * 5, fc_hidden * 5 // 2, in_fc_layer=True),
            nn.Dropout(p=0.75),
            nn.BatchNorm1d(fc_hidden * 5 // 2), #даёт ошибку если тестировать по одному файлу а не батч целиком - просто комментить для теста
            nn.Linear(fc_hidden * 5 // 2, n_class)
        )
    
    # def _computeLinearInputSize(self):

    def forward(self, x):
        out = self.net(x)
        return {"output": out}