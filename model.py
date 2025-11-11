import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_objectives import GCCA_loss

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 这里 Linear 输入固定为 512 * block.expansion，因为后面会池化成 1x1
        self.reshape = nn.Sequential(
            nn.Linear(512 * block.expansion, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 自适应池化到 1x1，避免不同输入尺寸带来的问题
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)  # (batch, 512 * expansion)
        out = self.reshape(out)
        return F.normalize(out)


class VGG(nn.Module):
    def __init__(self, vgg_name, feature_dim=128, num_classes=10, input_size=(3, 32, 32)):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        # 自动计算 reshape 输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            feat_out = self.features(dummy_input)
            feat_dim = feat_out.view(1, -1).size(1)

        self.reshape = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_feature=True):
        out = self.features(x)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        feature = self.reshape(out)
        feature = F.normalize(feature)
        if return_feature:
            return feature
        else:
            return self.classifier(feature)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                # 小输入直接跳过 MaxPool 或者用 1x1 自适应池化
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

# ============ Simple CNN Backbone ============
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, feature_dim=128, in_channels=1):
        super(SimpleCNN, self).__init__()
        
        # 卷积 backbone
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 每次减半

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # 自适应池化，保证输出维度固定
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接映射到 embedding
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_feature=True):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        feature = self.fc(x)

        feature = F.normalize(feature)  # 如果觉得容易塌缩，可以注释掉
        if return_feature:
            return feature
        else:
            return self.classifier(feature)

class Encoder(nn.Module):

    def __init__(self, dim_z, model):
        super().__init__()

        # print("DEBUG: modality is: ", modality)
        self.dim_z = dim_z
        if model == 'res18':
            self.encoder = ResNet(BasicBlock, [2, 2, 2, 2], self.dim_z)
        elif model == 'res34':
            self.encoder = ResNet(BasicBlock, [3, 4, 6, 3], self.dim_z)
        elif model == 'vgg11':
            self.encoder = VGG('VGG11', feature_dim=self.dim_z)
        elif model =='vgg16':
            self.encoder = VGG('VGG16', feature_dim=self.dim_z)
        elif model == 'simplecnn':
            self.encoder = SimpleCNN(feature_dim=self.dim_z, in_channels=1)

    def forward(self, x):
        # print(x.shape)
        feature = self.encoder(x)

        return feature

class Classifier(nn.Module):

    def __init__(self, num_classes, dim_z):
        super().__init__()

        # print("DEBUG: modality is: ", modality)
        
        self.dim_z = dim_z
        self.classifier = nn.Linear(self.dim_z, num_classes)

    def forward(self, feature):
        # print(x.shape)
        output = self.classifier(feature)

        return output


class DeepGCCA(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values=False, device=torch.device('cpu')):
        super(DeepGCCA, self).__init__()
        self.model_list = []
        for i in range(6):
            self.model_list.append(SimpleCNN(feature_dim=outdim_size, in_channels=1))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss

    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x)) 

        return output_list


