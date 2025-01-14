import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel, DataParallel

import logging
from utils import util
from collections import OrderedDict
from utils.JPEG import DiffJPEG
from models.modules.Inv_arch import InvNN, PredictiveModuleMIMO_prompt, DW_Encoder, DW_Decoder
from models.modules.Subnet_constructor import subnet
from models.modules.Quantization import Quantization

# for training
from models.modules.loss import ReconstructionLoss, ReconstructionMsgLoss
import models.lr_scheduler as lr_scheduler

# for robust method
from robust_method import robust_hidden, decode_hidden

# for debugging
from torchvision.utils import save_image

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

# Global variables
dwt=DWT()
iwt=IWT()
logger = logging.getLogger('base')

class VSN(nn.Module):
    def __init__(self, opt, subnet_constructor=None, subnet_constructor_v2=None, down_num=2):
        super(VSN, self).__init__()
        self.model = opt['model']
        self.mode = opt['mode']
        opt_net = opt['network_G']
        self.num_image = opt['num_image']
        self.gop = opt['gop']
        self.channel_in = opt_net['in_nc'] * self.gop
        self.channel_out = opt_net['out_nc'] * self.gop
        self.channel_in_hi = opt_net['in_nc'] * self.gop
        self.channel_in_ho = opt_net['in_nc'] * self.gop
        self.message_len = opt['message_length']

        self.block_num = opt_net['block_num']
        self.block_num_rbm = opt_net['block_num_rbm']
        self.block_num_trans = opt_net['block_num_trans']
        self.nf = self.channel_in_hi 
        
        self.bitencoder = DW_Encoder(self.message_len, attention = "se")
        self.bitdecoder = DW_Decoder(self.message_len, attention = "se")
        self.irn = InvNN(self.channel_in_ho, self.channel_in_hi, subnet_constructor, subnet_constructor_v2, self.block_num, down_num, groups=self.num_image)

        if opt['prompt']:
            self.pm = PredictiveModuleMIMO_prompt(self.channel_in_ho, self.nf* self.num_image, opt['prompt_len'], block_num_rbm=self.block_num_rbm, block_num_trans=self.block_num_trans)
        # else:
        #     self.pm = PredictiveModuleMIMO(self.channel_in_ho, self.nf* self.num_image, opt['prompt_len'], block_num_rbm=self.block_num_rbm, block_num_trans=self.block_num_trans)
        #     self.BitPM = PredictiveModuleBit(3, 4, block_num_rbm=4, block_num_trans=2)


    def forward(self, x, x_h=None, message=None, rev=False, hs=[], direction='f'):
        if not rev:
            # 我這邊意義跟原始paper不同
            # 原本先train bit mode，再train image mode時，會用到bit mode的encoder
            # 現在先train image mode，再train bit mode時，會用到image INN
            if self.mode == "image":
                out_y, out_y_h = self.irn(x, x_h, rev)
                out_y = iwt(out_y)
                # encoded_image = self.bitencoder(out_y, message)          
                # print("out_y", out_y.shape) # torch.Size([1, 3, 512, 512])
                # print(out_y.min(), out_y.max()) # [0, 1]

                
                encoded_image = out_y
                return out_y, encoded_image
            elif self.mode == "bit":
                original_image = iwt(x)
                encoded_image = self.bitencoder(original_image, message)            

                ######### ROI start #########
                # remove_ratio = 0.7 # this is ROI

                # center_size = int(512 * remove_ratio)
                # start_x = (512 - center_size) // 2
                # start_y = start_x

                # clear_center = original_image[:, :, start_x:start_x + center_size, start_y:start_y + center_size]

                # encoded_image[:, :, start_x:start_x + center_size, start_y:start_y + center_size] = clear_center
                ######### ROI end #########

                out_y, out_y_h = self.irn(dwt(encoded_image), x_h, rev)
                out_y = iwt(out_y)
                # print("encoded_image", encoded_image.shape) # torch.Size([1, 3, 512, 512])
                # print("out_y", out_y.shape)  # torch.Size([1, 3, 512, 512])
                # save_image(iwt(x), "gt.png")
                # save_image(iwt(x_h[0]), "se.png")
                # save_image(encoded_image, "rb.png")
                # save_image(out_y, "rbfg.png")
                return encoded_image, out_y
            
        else:
            if self.mode == "image":
                # recmessage = self.bitdecoder(x)
                recmessage = torch.Tensor(np.random.choice([-0.5, 0.5], (1, self.message_len))).to('cuda')

                x = dwt(x)
                out_z = self.pm(x).unsqueeze(1)
                out_z_new = out_z.view(-1, self.num_image, self.channel_in, x.shape[-2], x.shape[-1])
                out_z_new = [out_z_new[:,i] for i in range(self.num_image)]
                out_x, out_x_h = self.irn(x, out_z_new, rev)

                return out_x, out_x_h, out_z, recmessage
            
            elif self.mode == "bit":
                # 這邊INN inverse得到out_x, out_x_h不一定要做，但是為了debug我還是做了
                # save_image(x, "rbfg_protected.png")
                recmessage = self.bitdecoder(x)

                x = dwt(x)
                out_z = self.pm(x).unsqueeze(1)
                out_z_new = out_z.view(-1, self.num_image, self.channel_in, x.shape[-2], x.shape[-1])
                out_z_new = [out_z_new[:,i] for i in range(self.num_image)]
                out_x, out_x_h = self.irn(x, out_z_new, rev)
                # save_image(iwt(out_x), "cover_rec.png")
                # save_image(iwt(out_x_h[0]), "secret_rec.png")
                return out_x, out_x_h, recmessage
            
def define_G_v2(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    opt_datasets = opt['datasets']
    down_num = int(math.log(opt_net['scale'], 2))
    if opt['num_image'] == 1:
        netG = VSN(opt, subnet(subnet_type, 'xavier'), subnet(subnet_type, 'xavier'), down_num)
    else:
        netG = VSN(opt, subnet(subnet_type, 'xavier'), subnet(subnet_type, 'xavier_v2'), down_num)

    return netG

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        print('Loading model for G [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('bitencoder.') or k.startswith('bitdecoder.'):
                continue
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def load_bit_network(self, load_path, network, strict=True):
        print('Loading model for G [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('irn.') or k.startswith('pm.'):
                continue
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.num_image = opt['num_image']
        self.mode = opt["mode"]
        self.idxx = 0

        self.netG = define_G_v2(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        # self.print_network()
        self.load()

        self.Quantization = Quantization()

        # if not self.opt['hide']:
        #     file_path = "bit_sequence.txt"

        #     data_list = []

        #     with open(file_path, "r") as file:
        #         for line in file:
        #             data = [int(bit) for bit in line.strip()]
        #             data_list.append(data)
            
        #     self.msg_list = data_list

        # if self.opt['sdinpaint']:
        #     self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #         "stabilityai/stable-diffusion-2-inpainting",
        #         torch_dtype=torch.float16,
        #     ).to("cuda")
        
        # if self.opt['controlnetinpaint']:
        #     controlnet = ControlNetModel.from_pretrained(
        #         "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32
        #     ).to("cuda")
        #     self.pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        #         "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
        #     ).to("cuda")
        
        # if self.opt['sdxl']:
        #     self.pipe_sdxl = StableDiffusionXLInpaintPipeline.from_pretrained(
        #         "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        #         torch_dtype=torch.float16,
        #         variant="fp16",
        #         use_safetensors=True,
        #     ).to("cuda")
        
        # if self.opt['repaint']:
        #     self.scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
        #     self.pipe_repaint = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=self.scheduler)
        #     self.pipe_repaint = self.pipe_repaint.to("cuda")

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")
            self.Reconstruction_msg = ReconstructionMsgLoss(losstype=self.opt['losstype'])

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []

            if self.mode == "image":
                for k, v in self.netG.named_parameters():
                    if (k.startswith('module.irn') or k.startswith('module.pm')) and v.requires_grad: 
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            elif self.mode == "bit":
                for k, v in self.netG.named_parameters():
                    if (k.startswith('module.bitencoder') or k.startswith('module.bitdecoder')) and v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))


            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  
        self.real_H = data['GT'].to(self.device)
        # self.mes = data['MES']

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w]).cuda())
            c_t.append(torch.zeros([b, c, h, w]).cuda())
        memory = torch.zeros([b, c, h, w]).cuda()

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_back_rec(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec
    
    def loss_back_rec_mul(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec

    def optimize_parameters(self, current_step):
        self.optimizer_G.zero_grad()
      
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2

        message = torch.Tensor(np.random.choice([-0.5, 0.5], (self.ref_L.shape[0], self.opt['message_length']))).to(self.device)

        add_noise = self.opt['addnoise']
        add_jpeg = self.opt['addjpeg']
        add_possion = self.opt['addpossion']
        add_sdinpaint = self.opt['sdinpaint']
        degrade_shuffle = self.opt['degrade_shuffle']

        self.host = self.real_H[:, center - intval:center + intval + 1]
        self.secret = self.ref_L[:, :, center - intval:center + intval + 1]
        self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=dwt(self.secret[:,0].reshape(b, -1, h, w)), message=message)

        Gt_ref = self.real_H[:, center - intval:center + intval + 1].detach()

        y_forw = container

        if self.mode == "image":
            l_forw_fit = self.loss_forward(y_forw, self.host[:,0])
        elif self.mode == "bit":
            # 注意這裡應該要是self.output因為INN的output不用計算loss
            l_forw_fit = self.loss_forward(self.output, self.host[:,0])


        if degrade_shuffle:
            import random
            choice = random.randint(0, 2)
            
            if choice == 0:
                NL = float((np.random.randint(1, 16))/255)
                noise = np.random.normal(0, NL, y_forw.shape)
                torchnoise = torch.from_numpy(noise).cuda().float()
                y_forw = y_forw + torchnoise

            elif choice == 1:
                NL = int(np.random.randint(70,95))
                self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                y_forw = self.DiffJPEG(y_forw)
            
            elif choice == 2:
                vals = 10**4
                if random.random() < 0.5:
                    noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                else:
                    img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                    noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                    noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                y_forw = torch.clamp(noisy_img_tensor, 0, 1)

        else:

            if add_noise:
                NL = float((np.random.randint(1,16))/255)
                noise = np.random.normal(0, NL, y_forw.shape)
                torchnoise = torch.from_numpy(noise).cuda().float()
                y_forw = y_forw + torchnoise

            elif add_jpeg:
                NL = int(np.random.randint(70,95))
                self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                y_forw = self.DiffJPEG(y_forw)

            elif add_possion:
                vals = 10**4
                if random.random() < 0.5:
                    noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                else:
                    img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                    noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                    noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                y_forw = torch.clamp(noisy_img_tensor, 0, 1)

        y = self.Quantization(y_forw)
        all_zero = torch.zeros(message.shape).to(self.device)

        if self.mode == "image":
            out_x, out_x_h, out_z, recmessage = self.netG(x=y, message=all_zero, rev=True)
            out_x = iwt(out_x)
            out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]

            l_back_rec = self.loss_back_rec(out_x, self.host[:,0])
            out_x_h = torch.stack(out_x_h, dim=1)

            l_center_x = self.loss_back_rec(out_x_h[:, 0], self.secret[:,0].reshape(b, -1, h, w))

            recmessage = torch.clamp(recmessage, -0.5, 0.5)

            l_msg = self.Reconstruction_msg(message, recmessage)

            loss = l_forw_fit*2 + l_back_rec + l_center_x*4

            loss.backward()

            if self.train_opt['lambda_center'] != 0:
                self.log_dict['l_center_x'] = l_center_x.item()

            # set log
            self.log_dict['l_back_rec'] = l_back_rec.item()
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_msg'] = l_msg.item()
            
            self.log_dict['l_h'] = (l_center_x*10).item()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

        elif self.mode == "bit":
            out_x, out_x_h, recmessage = self.netG(x=y, message=all_zero, rev=True)

            recmessage = torch.clamp(recmessage, -0.5, 0.5)

            l_msg = self.Reconstruction_msg(message, recmessage)
            
            lambda_msg = self.train_opt['lambda_msg']

            loss = l_msg * lambda_msg + l_forw_fit

            loss.backward()

            # set log
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_msg'] = l_msg.item()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

    def test(self, image_id):
        self.netG.eval()
        add_noise = self.opt['addnoise']
        add_jpeg = self.opt['addjpeg']
        add_possion = self.opt['addpossion']
        add_sdinpaint = self.opt['sdinpaint']
        add_controlnet = self.opt['controlnetinpaint']
        add_sdxl = self.opt['sdxl']
        add_repaint = self.opt['repaint']
        degrade_shuffle = self.opt['degrade_shuffle']
        test_robust = self.opt['test_robust']

        with torch.no_grad():
            forw_L = []
            forw_L_h = []
            fake_H = []
            fake_H_h = []
            pred_z = []
            recmsglist = []
            msglist = []
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            # print(self.host.shape) # torch.Size([1, 1, 3, 512, 512])
            # print(self.secret.shape) # torch.Size([1, 1, 1, 3, 512, 512])
            self.secret = [self.secret[:,i].reshape(b, -1, h, w) for i in range(n)]
            # print(self.secret[0].shape) # torch.Size([1, 3, 512, 512])
            # print("Range of secret", self.secret[0].min(), self.secret[0].max())
            self.secret = [dwt(self.secret[i]) for i in range(n)]
            # print(self.secret[0].shape) # torch.Size([1, 12, 256, 256])
            # print("Range of secret after dwt", self.secret[0].min(), self.secret[0].max())

            messagenp = np.random.choice([-0.5, 0.5], (self.ref_L.shape[0], self.opt['message_length']))

            message = torch.Tensor(messagenp).to(self.device)

            if self.opt['bitrecord']:
                mymsg = message.clone()

                mymsg[mymsg>0] = 1
                mymsg[mymsg<0] = 0
                mymsg = mymsg.squeeze(0).to(torch.int)

                bit_list = mymsg.tolist()

                bit_string = ''.join(map(str, bit_list))

                file_name = "bit_sequence.txt"

                with open(file_name, "a") as file:
                    file.write(bit_string + "\n")

            if self.opt['hide']:
                self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=self.secret, message=message)
                # for visualization
                forw_L = []
                msglist = []
                recmsglist = []
                forw_L.append(container)
                self.forw_L = torch.clamp(torch.stack(forw_L, dim=1),0,1)
                self.rb_img = self.output
                y_forw = container
            else:
                
                message = torch.tensor(self.msg_list[image_id]).unsqueeze(0).cuda()
                self.output = self.host
                y_forw = self.output.squeeze(1)

            if add_sdinpaint:
                import random
                from PIL import Image
                prompt = ""

                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/"
                    mask_image = Image.open(masksrc + str(i).zfill(4) + ".png").convert("L")
                    mask_image = mask_image.resize((512, 512))
                    h, w = mask_image.size
                    
                    image = image_batch[j, :, :, :]
                    image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    image_inpaint = self.pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = np.array(mask_image)
                    mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            if add_controlnet:
                from diffusers.utils import load_image
                from PIL import Image

                b, _, _, _ = y_forw.shape
                forw_list = []
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                generator = torch.Generator(device="cuda").manual_seed(1)

                for j in range(b):
                    i = image_id + 1
                    mask_path = "../dataset/valAGE-Set-Mask/" + str(i).zfill(4) + ".png"
                    mask_image = load_image(mask_path)
                    mask_image = mask_image.resize((512, 512))
                    image_init = image_batch[j, :, :, :]
                    image_init1 = Image.fromarray((image_init * 255).astype(np.uint8), mode = "RGB")
                    image_mask = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

                    assert image_init.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
                    image_init[image_mask > 0.5] = -1.0  # set as masked pixel
                    image = np.expand_dims(image_init, 0).transpose(0, 3, 1, 2)
                    control_image = torch.from_numpy(image)

                    # generate image
                    image_inpaint = self.pipe_control(
                        "",
                        num_inference_steps=20,
                        generator=generator,
                        eta=1.0,
                        image=image_init1,
                        mask_image=image_mask,
                        control_image=control_image,
                    ).images[0]
                    
                    image_inpaint = np.array(image_inpaint) / 255.
                    image_mask = np.stack([image_mask] * 3, axis=-1)
                    image_mask = image_mask.astype(np.uint8)
                    image_fuse = image_init * (1 - image_mask) + image_inpaint * image_mask
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))

                y_forw = torch.stack(forw_list, dim=0).float().cuda()
            
            if add_sdxl:
                import random
                from PIL import Image
                from diffusers.utils import load_image
                prompt = ""

                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/"
                    mask_image = load_image(masksrc + str(i).zfill(4) + ".png").convert("RGB")
                    mask_image = mask_image.resize((512, 512))
                    h, w = mask_image.size
                    
                    image = image_batch[j, :, :, :]
                    image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    image_inpaint = self.pipe_sdxl(
                        prompt=prompt, image=image_init, mask_image=mask_image, num_inference_steps=50, strength=0.80, target_size=(512, 512)
                    ).images[0]
                    image_inpaint = image_inpaint.resize((512, 512))
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = np.array(mask_image) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            
            if add_repaint:
                from PIL import Image
                
                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                generator = torch.Generator(device="cuda").manual_seed(0)
                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/" + str(i).zfill(4) + ".png"
                    mask_image = Image.open(masksrc).convert("RGB")
                    mask_image = mask_image.resize((256, 256))
                    mask_image = Image.fromarray(255 - np.array(mask_image))
                    image = image_batch[j, :, :, :]
                    original_image = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    original_image = original_image.resize((256, 256))
                    output = self.pipe_repaint(
                        image=original_image,
                        mask_image=mask_image,
                        num_inference_steps=150,
                        eta=0.0,
                        jump_length=10,
                        jump_n_sample=10,
                        generator=generator,
                    )
                    image_inpaint = output.images[0]
                    image_inpaint = image_inpaint.resize((512, 512))
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = mask_image.resize((512, 512))
                    mask_image = np.array(mask_image) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * mask_image + image_inpaint * (1 - mask_image)
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            if degrade_shuffle:
                import random
                choice = random.randint(0, 2)
                
                if choice == 0:
                    NL = float((np.random.randint(1,5))/255)
                    noise = np.random.normal(0, NL, y_forw.shape)
                    torchnoise = torch.from_numpy(noise).cuda().float()
                    y_forw = y_forw + torchnoise

                elif choice == 1:
                    NL = 90
                    self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                    y_forw = self.DiffJPEG(y_forw)
                
                elif choice == 2:
                    vals = 10**4
                    if random.random() < 0.5:
                        noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                    else:
                        img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                        noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                        noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                    y_forw = torch.clamp(noisy_img_tensor, 0, 1)

            else:

                if add_noise:
                    NL = self.opt['noisesigma'] / 255.0
                    noise = np.random.normal(0, NL, y_forw.shape)
                    torchnoise = torch.from_numpy(noise).cuda().float()
                    y_forw = y_forw + torchnoise

                elif add_jpeg:
                    Q = self.opt['jpegfactor']
                    self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(Q)).cuda()
                    y_forw = self.DiffJPEG(y_forw)

                elif add_possion:
                    vals = 10**4
                    if random.random() < 0.5:
                        noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                    else:
                        img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                        noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                        noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                    y_forw = torch.clamp(noisy_img_tensor, 0, 1)

            # Test localization for inpainting
            # import random
            # massive_inpaint_ratio = 0.05
            # b, c, h, w = y_forw.shape
            # modified_h = int(h * massive_inpaint_ratio)
            # modified_w = int(w * massive_inpaint_ratio)
            # # sample the upper-left corner of the modified area
            # random_h = random.randint(0, h - modified_h)
            # random_w = random.randint(0, w - modified_w)
            # y_forw[:, :, random_h:random_h + modified_h, random_w:random_w + modified_w] = 0

            # 在這裡改成其他robust watermarking的方法
            if test_robust:
                encoded_image, recmessage = robust_hidden(y_forw, message)
                y_forw = encoded_image

            # backward upscaling
            if self.opt['hide']:
                y = self.Quantization(y_forw)
            else:
                y = y_forw

            if self.mode == "image":
                if test_robust:
                    out_x, out_x_h, out_z, _ = self.netG(x=y, rev=True)
                else:
                    out_x, out_x_h, out_z, recmessage = self.netG(x=y, rev=True)

                out_x = iwt(out_x)

                out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
                out_x = out_x.reshape(-1, self.gop, 3, h, w)
                out_x_h = torch.stack(out_x_h, dim=1)
                out_x_h = out_x_h.reshape(-1, 1, self.gop, 3, h, w)
                # 測試能不能直接從template中decode出bitstream
                # print(out_x_h.shape) # [1, 1, 1, 3, 512, 512]
                # decode_hidden(out_x_h[0,0])

                forw_L.append(y_forw)
                fake_H.append(out_x[:, self.gop//2])
                fake_H_h.append(out_x_h[:,:, self.gop//2])
                recmsglist.append(recmessage)
                msglist.append(message)
            
            elif self.mode == "bit":
                # for visualization
                fake_H = []
                fake_H_h = []
                out_x, out_x_h, recmessage = self.netG(x=y, rev=True)
                out_x = iwt(out_x)

                out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
                out_x = out_x.reshape(-1, self.gop, 3, h, w)
                out_x_h = torch.stack(out_x_h, dim=1)
                out_x_h = out_x_h.reshape(-1, 1, self.gop, 3, h, w)

                fake_H.append(out_x[:, self.gop//2])
                fake_H_h.append(out_x_h[:,:, self.gop//2])
                self.fake_H = torch.clamp(torch.stack(fake_H, dim=1),0,1)
                self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=2),0,1) 


                forw_L.append(y_forw)
                recmsglist.append(recmessage)
                msglist.append(message)

        if self.mode == "image":
            self.fake_H = torch.clamp(torch.stack(fake_H, dim=1),0,1)
            self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=2),0,1)

        self.forw_L = torch.clamp(torch.stack(forw_L, dim=1),0,1)
        remesg = torch.clamp(torch.stack(recmsglist, dim=0),-0.5,0.5)

        if self.opt['hide']:
            mesg = torch.clamp(torch.stack(msglist, dim=0),-0.5,0.5)
        else:
            mesg = torch.stack(msglist, dim=0)

        self.recmessage = remesg.clone()
        self.recmessage[remesg > 0] = 1
        self.recmessage[remesg <= 0] = 0

        self.message = mesg.clone()
        self.message[mesg > 0] = 1
        self.message[mesg <= 0] = 0

        self.netG.train()


    def image_hiding(self, ):
        self.netG.eval()
        with torch.no_grad():
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]

            message = torch.Tensor(self.mes).to(self.device)

            self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=self.secret, message=message)
            y_forw = container

            result = torch.clamp(y_forw,0,1)

            lr_img = util.tensor2img(result)

            return lr_img

    def image_recovery(self, number):
        self.netG.eval()
        with torch.no_grad():
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            template = self.secret.reshape(b, -1, h, w)
            self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]

            self.output = self.host
            y_forw = self.output.squeeze(1)

            y = self.Quantization(y_forw)

            out_x, out_x_h, out_z, recmessage = self.netG(x=y, rev=True)
            out_x = iwt(out_x)

            out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
            out_x = out_x.reshape(-1, self.gop, 3, h, w)
            out_x_h = torch.stack(out_x_h, dim=1)
            out_x_h = out_x_h.reshape(-1, 1, self.gop, 3, h, w)

            rec_loc = out_x_h[:,:, self.gop//2]
            # from PIL import Image
            # tmp = util.tensor2img(rec_loc)
            # save
            residual = torch.abs(template - rec_loc)
            binary_residual = (residual > number).float()
            residual = util.tensor2img(binary_residual)
            mask = np.sum(residual, axis=2)
            # print(mask)

            remesg = torch.clamp(recmessage,-0.5,0.5)
            remesg[remesg > 0] = 1
            remesg[remesg <= 0] = 0

            return mask, remesg
        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2
        out_dict = OrderedDict()
        LR_ref = self.ref_L[:, :, center - intval:center + intval + 1].detach()[0].float().cpu()
        LR_ref = torch.chunk(LR_ref, self.num_image, dim=0)
        out_dict['LR_ref'] = [image.squeeze(0) for image in LR_ref]
        
        if self.mode == "image":
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
            SR_h = self.fake_H_h.detach()[0].float().cpu()
            SR_h = torch.chunk(SR_h, self.num_image, dim=0)
            out_dict['SR_h'] = [image.squeeze(0) for image in SR_h]
        elif self.mode == "bit":
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
            SR_h = self.fake_H_h.detach()[0].float().cpu()
            SR_h = torch.chunk(SR_h, self.num_image, dim=0)
            out_dict['SR_h'] = [image.squeeze(0) for image in SR_h]
            out_dict['RB'] = self.rb_img.detach().float().cpu()
        
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H[:, center - intval:center + intval + 1].detach()[0].float().cpu()
        out_dict['message'] = self.message
        out_dict['recmessage'] = self.recmessage

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        else:
            print('Pre-trained model not found, initialize model from scratch.')
    
    def load_test(self,load_path_G):
        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
    def load_test_bit(self,load_path_G):
        self.load_bit_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)