"""
@author: Md. Rezwanul Haque

"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os 
import torch
from .config import CONFIG

### call CONFIG class
config = CONFIG()
torch.manual_seed(config.SEED)
os.environ["CUDA_VISIBLE_DEVCIES"] = config.GPU
use_gpu = torch.cuda.is_available()
if config.USE_CPU: use_gpu = False

"""
    * pre-trained ResNet
"""

class ResNet(nn.Module):
    """
        extracting features from images
        args:
            fea_type: string, resnet101 or resnet 152
    """

    def __init__(self, fea_type = 'resnet152'):
        super(ResNet, self).__init__()
        self.fea_type = fea_type
        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if fea_type == 'resnet101':
            resnet = models.resnet101(pretrained=True)  # dim of pool5 is 2048
        elif fea_type == 'resnet152':
            resnet = models.resnet152(pretrained=True)
        else:
            raise Exception('No such ResNet!')

        if use_gpu:
            resnet = resnet.cuda()
        resnet.float()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    # rescale and normalize image, then pass it through ResNet
    def forward(self, x):

        x = self.transform(x)
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim

        if use_gpu:
            x = Variable(x.cuda())   
        else:
            x = Variable(x)
        
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)

        return res_pool5