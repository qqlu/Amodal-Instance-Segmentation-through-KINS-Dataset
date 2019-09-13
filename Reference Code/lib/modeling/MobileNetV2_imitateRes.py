import torch.nn as nn
import math
import torch
import sys
sys.path.append('/media/qqlu/883A1C2C3A1C1A30/github/Detectron.pytorch-master/lib')
from core.config import cfg
import nn as mynn
# cfg.RESNETS.FREEZE_AT=1
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        mynn.AffineChannel2d(oup),
        # nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        mynn.AffineChannel2d(oup),
        # nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            mynn.AffineChannel2d(inp * expand_ratio),
            # nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            mynn.AffineChannel2d(inp * expand_ratio),
            # nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            mynn.AffineChannel2d(oup),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_body(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_body, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.block_flag = [False, True, True, False, True, False, True]
        self.convX = 5
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.features = [conv_bn(3, input_channel, 2)]
        self.feature1 = conv_bn(3, input_channel, 2)
        # building inverted residual blocks
        features = []
        split_index = []
        count = 0
        for setting, flag in zip(self.interverted_residual_setting, self.block_flag):
            t, c, n, s = setting
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
                count += 1
            if flag==True:
                split_index.append(count)
        self.num_layers = count
        self.split_index = split_index

        self.feature2 = features[0:split_index[0]]
        self.feature2 = nn.Sequential(*self.feature2)

        self.feature3 = features[split_index[0]:split_index[1]]
        self.feature3 = nn.Sequential(*self.feature3)

        self.feature4 = features[split_index[1]:split_index[2]]
        self.feature4 = nn.Sequential(*self.feature4)

        self.feature5 = features[split_index[2]:split_index[3]]
        self.feature5.append(conv_1x1_bn(input_channel, self.last_channel))
        self.feature5 = nn.Sequential(*self.feature5)
        
        self.spatial_scale = 1 / 32;
        self.dim_out = 1280
        self._init_modules()
        # self.mapping, self.orphans = self.detectron_weight_mapping()

    def train(self, mode=True):
        self.training = mode
        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'feature%d' % i).train(mode)

        # for i in range(cfg.RESNETS.FREEZE_AT+1, 5):
        #     getattr(self, 'feature%d'%i).train(mode)

    def forward(self, x):
        for i in range(1, self.convX+1):
            x = getattr(self, 'feature%d'%(i+1))(x)
        return x


    def _init_modules(self):
        for i in range(1, cfg.RESNETS.FREEZE_AT+1):
            freeze_params(getattr(self, 'feature%d'%i))
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)
        # self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel) else None)


    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            None
        else:
            mapping_to_detectron = {'feature1.0.weight': 'module.features.0.0.weight',
                                    'feature1.1.weight': 'module.features.0.1.weight',
                                    'feature1.1.bias': 'module.features.0.1.bias'}
            orphan_in_detectron =[]
        # print(self.split_index)
        for res_id in range(2, self.convX + 1):
            stage_name = 'feature%d' % res_id
            if res_id == 2:
                start = 0
                end = self.split_index[res_id-2]
            else:
                # print(res_id)
                start = self.split_index[res_id-3]
                end = self.split_index[res_id-2]

            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, start, end
                )
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)
        mapping_to_detectron.update({'feature5.4.0.weight': 'module.features.18.0.weight', 
                                    'feature5.4.1.weight': 'module.features.18.1.weight',
                                    'feature5.4.1.bias': 'module.features.18.1.bias'})
        return mapping_to_detectron, orphan_in_detectron


def residual_stage_detectron_mapping(module_ref, module_name, start, end):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    mapping_to_detectron = {}
    orphan_in_detectron = []
    key_id = 0
    for idx in range(start, end): 
        value_id = idx + 1
        # print(idx)
        key_prefix = '.'.join([module_name, str(key_id)])
        value_prefix = '.'.join(['module', 'features', str(value_id), 'conv'])
        
        mapping_to_detectron[key_prefix+'.conv.0.weight'] = value_prefix+'.0.weight'
        mapping_to_detectron[key_prefix+'.conv.1.weight'] = value_prefix+'.1.weight'
        mapping_to_detectron[key_prefix+'.conv.1.bias'] = value_prefix+'.1.bias'
        mapping_to_detectron[key_prefix+'.conv.3.weight'] = value_prefix+'.3.weight'
        mapping_to_detectron[key_prefix+'.conv.4.weight'] = value_prefix+'.4.weight'
        mapping_to_detectron[key_prefix+'.conv.4.bias'] = value_prefix+'.4.bias'
        mapping_to_detectron[key_prefix+'.conv.6.weight'] = value_prefix+'.6.weight'
        mapping_to_detectron[key_prefix+'.conv.7.weight'] = value_prefix+'.7.weight'
        mapping_to_detectron[key_prefix+'.conv.7.bias'] = value_prefix+'.7.bias'
        key_id += 1

    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    for p in m.parameters():
        p.requires_grad = False

if __name__ == '__main__':
    net = MobileNetV2_body(n_class=1000)
    # print(net.mapping)
    # print('------net------')
    # print(net.state_dict().keys())
    # net = torch.nn.DataParallel(net).cuda()
    state_dict = torch.load('/media/qqlu/883A1C2C3A1C1A30/github/Detectron.pytorch-master/data/pretrained_model/mobilenetv2_718.pth.tar')
    net_state_dict = net.state_dict()
    for key, value in net.mapping.items():
        net_state_dict[key] = state_dict[value]
    print('success')
    # print('------ckpt------')
    # print(state_dict.keys())
    # net.load_state_dict(state_dict)    
