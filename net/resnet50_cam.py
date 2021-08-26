import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnext as resnet50
import torch
from net import Gaussian

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x ,x_ref):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1) #b, hw, c


        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  #b, hw, c

        phi_x = self.phi(x_ref).view(batch_size, self.inter_channels, -1) #b, c ,hw
        f = torch.matmul(theta_x, phi_x) #b, hw,hw
        N = f.size(-1)
        f_div_C = f / N

        # print(f_div_C.shape)

        y = torch.matmul(f_div_C, g_x) #b,hw,c
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

mid = 512

def cal_cos_vector(x,mask):
    h, w = mask.shape[-2:][0], mask.shape[-2:][1]
    area = F.avg_pool2d(mask, x.shape[-2:]) * h * w + 0.0005
    z = mask * x
    z = F.avg_pool2d(input=z,
                     kernel_size=x.shape[-2:]) * h * w / area
    return z

def cal_cos_vector_part(x,mask):
    mask = F.dropout(mask,p=0.5) ## dropout will increase the number
    mask = (mask / mask.max()) ## make sure 1

    h, w = mask.shape[-2:][0], mask.shape[-2:][1]
    area = F.avg_pool2d(mask, x.shape[-2:]) * h * w + 0.0005
    z = mask * x
    z = F.avg_pool2d(input=z,
                     kernel_size=x.shape[-2:]) * h * w / area
    return z

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnext101_32x8d(pretrained=True)

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.mask_t = nn.Conv2d(mid, 128, 1, stride=2, bias=False)
        self.mask = nn.Conv2d(mid,1,1,stride=2,bias=False)

        self.classifierx = nn.Conv2d(mid, 20, 1, bias=False)
        self.classifiery = nn.Conv2d(mid, 20, 1, bias=False)

        self.cross_atten_x = NONLocalBlock2D(mid)
        self.cross_atten_y = NONLocalBlock2D(mid)
        self.cross_atten   = NONLocalBlock2D(mid * 2)

        self.self_atten_x = NONLocalBlock2D(mid)
        self.self_atten_y = NONLocalBlock2D(mid)

        self.pre_x = nn.Sequential(nn.Conv2d(mid*2,mid,kernel_size=3,padding=1),
                                   nn.ReLU(),
                                   nn.Dropout2d(p=0.5)
        )

        self.pre_y = nn.Sequential(nn.Conv2d(mid*2, mid, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Dropout2d(p=0.5)
                                   )

        self.downsample = nn.Conv2d(2048, mid, 1, bias=False)

        self.gauss = Gaussian.GaussianSmoothing(1,5,3)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifierx,self.classifiery,self.mask_t,self.cross_atten,
                                          self.cross_atten_x,self.cross_atten_y,self.self_atten_x,
                                          self.self_atten_y,self.pre_x,self.pre_y,self.mask,self.downsample])

    def forward(self, x,y):
        x = self.stage1(x)
        x2 = self.stage2(x).detach()
        x3 = self.stage3(x2)
        x = self.stage4(x3)

        y = self.stage1(y)
        y2 = self.stage2(y).detach()
        y3 = self.stage3(y2)
        y = self.stage4(y3)

        x = self.downsample(x)
        y = self.downsample(y)

        x_s = self.self_atten_x(x,x)
        y_s = self.self_atten_y(y,y)

        support_x = (x / (F.adaptive_max_pool2d(x, (1, 1)) + 1e-5))  # Normal
        support_y = (y / (F.adaptive_max_pool2d(y, (1, 1)) + 1e-5))

        ## Define foreground mask
        support_x_fg = torch.ge(support_x, 0.3) * 1.
        support_x_fg, _ = torch.max(support_x_fg, dim=1, keepdim=True)

        support_y_fg = torch.ge(support_y, 0.3) * 1.
        support_y_fg, _ = torch.max(support_y_fg, dim=1, keepdim=True)

        support_x_mask_fg, support_y_mask_fg = self.cal_mask(x,y,support_x_fg,support_y_fg)

        x2 = F.upsample_bilinear(x2,size=x.shape[-2:])
        y2 = F.upsample_bilinear(y2,size=y.shape[-2:])
        
        support_x_mask_fg2, support_y_mask_fg2 = self.cal_mask(x2, y2, support_x_fg, support_y_fg)

        support_x_mask_fg = torch.max(support_x_mask_fg, support_x_mask_fg2)
        support_y_mask_fg = torch.max(support_y_mask_fg, support_y_mask_fg2)

        x_mask = (support_x_mask_fg / (F.adaptive_max_pool2d(support_x_mask_fg, (1, 1)) + 1e-5))  * 1.3
        y_mask = (support_y_mask_fg / (F.adaptive_max_pool2d(support_y_mask_fg, (1, 1)) + 1e-5))  * 1.3

        x_mask = self.gauss(x_mask) * x
        y_mask = self.gauss(y_mask) * y

        x_mask = self.cross_atten_x(x_mask,x_mask)
        y_mask = self.cross_atten_y(y_mask,y_mask)

        x_mask = self.classifierx(x_mask)
        y_mask = self.classifiery(y_mask)

        x = self.classifierx(x_s)
        y = self.classifiery(y_s)
        return x,y,x_mask,y_mask

    def cal_mask(self,x,y,support_x_fg,support_y_fg):
        B, C, H, W = x.shape

        cos_x_fg = cal_cos_vector(x, support_x_fg)
        cos_y_fg = cal_cos_vector(y, support_y_fg)

        cos_x_fg_part = cal_cos_vector_part(x, support_x_fg)
        cos_y_fg_part = cal_cos_vector_part(y, support_y_fg)

        support_x_mask_fg = F.cosine_similarity(x, cos_y_fg).view(B, 1, H, W)  # B * H * W
        support_x_mask_fg_self = F.cosine_similarity(x, cos_x_fg).view(B, 1, H, W)
        support_x_mask_fg = torch.max(support_x_mask_fg, support_x_mask_fg_self)

        support_x_mask_fg_part = F.cosine_similarity(x, cos_y_fg_part).view(B, 1, H, W)  # B * H * W
        support_x_mask_fg_self_part = F.cosine_similarity(x, cos_x_fg_part).view(B, 1, H, W)
        support_x_mask_fg_part = torch.max(support_x_mask_fg_part, support_x_mask_fg_self_part)

        support_x_mask_fg = torch.max(support_x_mask_fg,support_x_mask_fg_part)

        support_y_mask_fg = F.cosine_similarity(y, cos_x_fg).view(B, 1, H, W)  # B * H * W
        support_y_mask_fg_self = F.cosine_similarity(y, cos_y_fg).view(B, 1, H, W)
        support_y_mask_fg = torch.max(support_y_mask_fg , support_y_mask_fg_self)

        support_y_mask_fg_part = F.cosine_similarity(y, cos_x_fg_part).view(B, 1, H, W)  # B * H * W
        support_y_mask_fg_self_part = F.cosine_similarity(y, cos_y_fg_part).view(B, 1, H, W)
        support_y_mask_fg_part = torch.max(support_y_mask_fg_part , support_y_mask_fg_self_part)

        support_y_mask_fg = torch.max(support_y_mask_fg,support_y_mask_fg_part)

        return support_x_mask_fg, support_y_mask_fg

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x,y):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.downsample(x)
        x_s = self.self_atten_x(x, x)
        x = x_s
        x = F.conv2d(x, self.classifierx.weight)
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

class Net_CAM(Net):
    def __init__(self):
        super().__init__()
    def forward(self, x,y):
        input_size_h = x.size()[2]
        input_size_w = x.size()[3]

        x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear',
                           align_corners=False)
        x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',
                           align_corners=False)
        x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',
                           align_corners=False)

        y2 = F.interpolate(y, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear',
                           align_corners=False)
        y3 = F.interpolate(y, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',
                           align_corners=False)
        y4 = F.interpolate(y, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',
                           align_corners=False)

        x_cam, y_cam, x_mask, y_mask = super().forward(x,y)
        with torch.no_grad():
            x2, y2, x_mask2, y_mask2 = super().forward(x2, y2)
            x3, y3, x_mask3, y_mask3 = super().forward(x3, y3)
            x4, y4, x_mask4, y_mask4 = super().forward(x4, y4)

        x_mask2 = F.interpolate(x_mask2, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear', align_corners=False)
        x_mask3 = F.interpolate(x_mask3, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear', align_corners=False)
        x_mask4 = F.interpolate(x_mask4, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear', align_corners=False)

        y_mask2 = F.interpolate(y_mask2, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear',
                                align_corners=False)
        y_mask3 = F.interpolate(y_mask3, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear',
                                align_corners=False)
        y_mask4 = F.interpolate(y_mask4, size=(int(x_mask.shape[2]), int(x_mask.shape[3])), mode='bilinear',
                                align_corners=False)

        cam_x = (F.relu(x_mask) + F.relu(x_mask2) +F.relu(x_mask3) +F.relu(x_mask4) )/4
        cam_y = (F.relu(y_mask) + F.relu(y_mask2) + F.relu(y_mask3) + F.relu(y_mask4)) / 4

        x = torchutils.gap2d(x_cam, keepdims=True)
        x = x.view(-1, 20)

        y = torchutils.gap2d(y_cam, keepdims=True)
        y = y.view(-1, 20)

        return x,y,x_cam,y_cam,cam_x,cam_y
