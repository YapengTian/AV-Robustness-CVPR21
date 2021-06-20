import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.misc import imsave

# Our libs
from arguments import ArgParser
from dataset import MUSICDataset
from nets import ModelBuilder, activate

from utils import  AverageMeter,makedirs
from viz import plot_loss_metrics,HTMLVisualizer
import matplotlib.pyplot as plt



def norm_tensor(x):
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    for i in range(x.size(0)):
        x[i, :, :, :] = norm(x[i])
    return x

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def inv_norm_tensor(x):
    inv_norm = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    for i in range(x.size(0)):
        x[i, :, :, :] = inv_norm(x[i])

    return x

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, args, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_classifier = nets
        self.D_v = args.Dv
        self.D_a = args.Da
        self.I = args.I

    def compute_coef(self, D, X):
        #A = torch.softmax(torch.mm(D.t().to(X.device), X)/np.sqrt(512), dim=0)#
        A = torch.mm(torch.mm(torch.inverse(torch.mm(D.t().to(X.device), D.to(X.device))+0.1*self.I.to(X.device)), D.t().to(X.device)), X)

        return A

    def compute_couple_coef(self, Da, Xa, Dv, Xv):
        A = torch.mm(torch.inverse(torch.mm(Da.t().to(Xa.device), Da.to(Xa.device)) + torch.mm(Dv.t().to(Xv.device), Dv.to(Xv.device))
                                            +0.1*self.I.to(Xa.device)), torch.mm(Da.t().to(Xa.device), Xa) + torch.mm(Dv.t().to(Xv.device), Xv))
        return A

    def shrink(self, x, alpha):
        return torch.mul(torch.sign(x), torch.clamp(torch.abs(x) - alpha, min=0.))

    def ISTA(self, D, x, h, alpha, lamb):
        critical_value = (torch.eig(torch.mm(D.t().to(x.device), D.to(x.device)))[0][:, 0]).max()

        if alpha * critical_value > 1:
            raise ValueError('Your alpha is too big for ISTA to converge.')
        converged = False
        iter = 0
        while not converged:
            h_prime = h
            h = h - torch.mm(alpha * D.t().to(x.device), torch.mm(D.to(x.device), h.to(x.device)) - x)
            h = self.shrink(h, alpha * lamb)
            # h = torch.clamp(h, min=0) ## every iter clamp negative values to 0
            if torch.norm(h - h_prime) < 0.0008:
                converged = True
            iter += 1
            if iter>2000: #2000
                break
        return h

    def forward(self, frame, audio):
        feat_frame = self.net_frame(frame)
        feat_sound = self.net_sound(audio)
        feat_sound = activate(feat_sound, args.sound_activation)
        feat_frame = activate(feat_frame, args.img_activation)

        # use dic
        feat_sound = feat_sound.permute(1,0)
        feat_frame = feat_frame.permute(1,0)
        # A_v = self.compute_coef(self.D_v, feat_frame)
        # A_a = self.compute_coef(self.D_a, feat_sound)

        A_v = self.ISTA(self.D_v, feat_frame, torch.zeros(self.D_v.size(1), feat_frame.size(1)).to(feat_frame.device), alpha=6e-6, lamb=0.1) #MUSIC 6e-6/kINETICS2e-6/ave 2e-6
        A_a = self.ISTA(self.D_a, feat_sound,torch.zeros(self.D_v.size(1), feat_frame.size(1)).to(feat_frame.device), alpha=1e-6, lamb=0.1) #MUSIC 1e-6/kINETICS5.8e-7/ave7e-7

        a = 0.5 # 0.6 KS 0.5MUSIC
        feat_frame= torch.mm(self.D_v.to(feat_frame.device), A_v).permute(1,0)*a +  (1-a)*feat_frame.permute(1,0)#
        feat_sound = torch.mm(self.D_a.to(feat_sound.device), A_a).permute(1,0)*a +  (1-a)*feat_sound.permute(1,0)#
        pred = self.net_classifier(feat_frame, feat_sound)

        return pred, feat_frame, feat_sound


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    criterion = nn.CrossEntropyLoss()
    torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    correct = 0

    total = 0
    for i, batch_data in enumerate(loader):
        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']

        audio = audios[0].to(args.device).detach()
        frame = frames[0].to(args.device).squeeze(2).detach()
        gt = gts[0].to(args.device)
        # netWrapper.zero_grad()
        # forward pass
        preds, feat_v, feat_a = netWrapper(frame, audio)
        err = criterion(preds, gt) #+ F.cosine_similarity(feat_v, feat_a, 1).mean()

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        loss_meter.update(err.item())
        # print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))
    acc = 100 * correct / total
    print('[Eval Summary] Epoch: {}, Loss: {:.4f}'
          .format(epoch, loss_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc)
    print('Accuracy of the audio-visual event recognition network: %.2f %%' % (
            100 * correct / total))

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


def evaluate_adv(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    criterion = nn.CrossEntropyLoss()
    # torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()
    # initialize meters
    loss_meter = AverageMeter()

    fig = plt.figure()
    epsilons = []
    for i in range(5):
        for j in range(5):
            epsilons.append([i*0.001, j*0.001])
    ep = 0.006
    epsilons = [[0, 0], [0, ep], [ep, 0], [ep, ep]]
    for epsilon in epsilons:
        cos_sim = []
        # initialize HTML header
        visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
        header = ['Filename']
        for n in range(1, args.num_mix + 1):
            header += ['Original Image {:d}'.format(n),
                       'Adv. Image {:d}'.format(n),
                       'Original Audio {}'.format(n),
                       'Adv. Audio {}'.format(n)]
        visualizer.add_header(header)
        vis_rows = []
        correct = 0
        adv_correct = 0
        total = 0

        for i, batch_data in enumerate(loader):
            audios = batch_data['audios']
            frames = batch_data['frames']
            gts = batch_data['labels']


            audio = audios[0].to(args.device)
            frame = frames[0].to(args.device).squeeze(2)
            gt = gts[0].to(args.device)

            if args.attack_type == "fsgm":
                data_viz = []
                frame.requires_grad = True
                audio.requires_grad = True

                # forward pass
                preds, feat_v, feat_a = netWrapper(frame, audio)
                netWrapper.zero_grad()
                err = criterion(preds, gt) + F.cosine_similarity(feat_v, feat_a, 1).mean() #0.8-ks
                err.backward()

                # original frame and audio
                frame_ori = inv_norm_tensor(frame.clone())
                data_viz.append(frame_ori)
                data_viz.append(audio)

                # Add perturbation
                if args.arch_classifier != "audio":
                    frame_adv = frame + epsilon[0] * torch.sign(frame.grad.data)
                else:
                    frame_adv = frame
                frame_adv = inv_norm_tensor(frame_adv.clone())

                frame_adv = torch.clamp(frame_adv, 0, 1)
                data_viz.append(frame_adv)
                frame_adv = norm_tensor(frame_adv.clone())

                if args.arch_classifier != "visual" :
                    audio_adv = audio + epsilon[1]  * torch.sign(audio.grad.data)
                    # audio_adv = torch.clamp(audio_adv, -1, 1).detach()
                else:
                    audio_adv = audio
                data_viz.append(audio_adv)

                adv_preds, feat_v, feat_a = netWrapper(frame_adv,audio_adv)
                sim = F.cosine_similarity(feat_v, feat_a, -1)
                cos_sim = np.concatenate((cos_sim, sim.detach().cpu().numpy()), axis=0)
            elif args.attack_type == "pgd":
                # original frame and audio
                data_viz = []
                frame_ori = inv_norm_tensor(frame.clone())
                data_viz.append(frame_ori)
                data_viz.append(audio)
                preds, _, _ = netWrapper(frame, audio)
                alpha_v = epsilon[0] / 8
                alpha_a = epsilon[1] / 8
                frame_adv = frame.clone().detach()
                audio_adv = audio.clone().detach()
                for t in range(10):
                    frame_adv.requires_grad = True
                    audio_adv.requires_grad = True
                    # forward pass
                    preds_iter, feat_v, feat_a = netWrapper(frame_adv, audio_adv)
                    netWrapper.zero_grad()
                    err = criterion(preds_iter, gt) + F.cosine_similarity(feat_v, feat_a, 1).mean()
                    err.backward()

                    # Add perturbation
                    if args.arch_classifier in ["concat", "visual"]:
                        frame_adv = frame_adv.detach() + alpha_v * torch.sign(frame_adv.grad.data)
                        eta = torch.clamp(frame_adv - frame, min=-epsilon[0], max=epsilon[0])
                        frame_adv = (frame + eta).detach_()
                    else:
                        frame_adv = frame.detach()
                    frame_adv = inv_norm_tensor(frame_adv.clone())
                    frame_adv = torch.clamp(frame_adv, 0, 1)
                    frame_adv = norm_tensor(frame_adv.clone())

                    if args.arch_classifier in ["concat", "audio"]:
                        audio_adv = audio_adv.detach() + alpha_a * torch.sign(audio_adv.grad.data)
                        eta = torch.clamp(audio_adv - audio, min=-epsilon[1], max=epsilon[1])
                        audio_adv = torch.clamp(audio + eta, min=-1, max=1).detach_()
                    else:
                        audio_adv = audio.detach()

                data_viz.append(torch.clamp(inv_norm_tensor(frame_adv.clone()), 0, 1))
                data_viz.append(audio_adv)
                adv_preds, _, _ = netWrapper(frame_adv, audio_adv)
            elif args.attack_type == "mim":
                # original frame and audio
                data_viz = []
                frame_ori = inv_norm_tensor(frame.clone())
                data_viz.append(frame_ori)
                data_viz.append(audio)
                preds, _, _ = netWrapper(frame, audio)

                alpha_v = epsilon[0]/8
                alpha_a = epsilon[1]/8
                frame_adv = frame.clone().detach()
                audio_adv = audio.clone().detach()
                momentum_v = torch.zeros_like(frame).to(args.device)
                momentum_a = torch.zeros_like(audio).to(args.device)

                for t in range(10):
                    frame_adv.requires_grad = True
                    audio_adv.requires_grad = True
                    # forward pass
                    preds_iter, feat_v, feat_a = netWrapper(frame_adv, audio_adv)
                    netWrapper.zero_grad()
                    err = criterion(preds_iter, gt) + F.cosine_similarity(feat_v, feat_a, 1).mean()
                    err.backward()

                    # Add perturbation
                    if args.arch_classifier in ["concat", "visual"]:
                        grad = frame_adv.grad.data
                        grad_norm = torch.norm(grad, p=1)
                        grad /= grad_norm
                        grad += momentum_v * 1.0
                        momentum_v = grad
                        frame_adv = frame_adv.detach() + alpha_v * torch.sign(grad)
                        a = torch.clamp(frame_adv - epsilon[0], min=0)
                        b = (frame_adv >= a).float() * frame_adv + (a > frame_adv).float() * a
                        c = (b > frame_adv + epsilon[0]).float() * (frame_adv + epsilon[0]) + (
                                frame_adv + epsilon[0] >= b
                        ).float() * b
                        frame_adv = c.detach_()
                    else:
                        frame_adv = frame.detach()
                    frame_adv = inv_norm_tensor(frame_adv.clone())
                    frame_adv = torch.clamp(frame_adv, 0, 1)
                    frame_adv = norm_tensor(frame_adv.clone())

                    if args.arch_classifier in ["concat", "audio"]:
                        grad = audio_adv.grad.data
                        grad_norm = torch.norm(grad, p=1)
                        grad /= grad_norm
                        grad += momentum_a * 1.0
                        momentum_a = grad
                        audio_adv = audio_adv.detach() + alpha_a * torch.sign(grad)
                        a = torch.clamp(audio_adv - epsilon[1], min=-1)
                        b = (audio_adv >= a).float() * audio_adv + (a > audio_adv).float() * a
                        c = (b > audio_adv + epsilon[1]).float() * (audio_adv + epsilon[1]) + (
                                audio_adv + epsilon[1] >= b
                        ).float() * b
                        audio_adv = c.detach_()
                        audio_adv = torch.clamp(audio_adv, min=-1, max=1).detach_()
                    else:
                        audio_adv = audio.detach()

                data_viz.append(torch.clamp(inv_norm_tensor(frame_adv.clone()), 0, 1))
                data_viz.append(audio_adv)
                adv_preds, _, _ = netWrapper(frame_adv, audio_adv)
            else:
                print("Unknown attack method!")

            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == gt).sum().item()

            _, predicted = torch.max(adv_preds.data, 1)
            adv_correct += (predicted == gt).sum().item()

            loss_meter.update(err.item())
            # print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

            # viz
            output_visuals(vis_rows,batch_data, data_viz, args)

        print('[Eval Summary] Epoch: {}, Loss: {:.4f}'
              .format(epoch, loss_meter.average()))
        history['val']['epoch'].append(epoch)
        history['val']['err'].append(loss_meter.average())

        print('Accuracy of the audio-visual event recognition network: %.2f %%' % (
                100 * correct / total))
        print('adv Accuracy of the audio-visual event recognition network: %.2f %%' % (
                100 * adv_correct / total))

        print('Plotting html for visualization...')
        visualizer.add_rows(vis_rows)
        visualizer.write_html()

        # Plot figure
        if epoch > 0:
            print('Plotting figures...')
            plot_loss_metrics(args.ckpt, history)

        plt.plot(cos_sim, label="v: "+ str(epsilon[0]*1e3)+ " + a: "+str(epsilon[1]*1e3)+" " + "acc: " + '%.1f'% (
                100 * adv_correct / total))
    plt.legend()
    fig.savefig(os.path.join(args.ckpt, 'cos_sim.png'), dpi=200)


# Visualize predictions
def output_visuals(vis_rows, batch_data, data_viz, args):

    infos = batch_data['infos']
    # fetch data and predictions
    frame_ori = data_viz[0]
    frame_adv = data_viz[2]
    audio_ori = data_viz[1]
    audio_adv = data_viz[3]

    B = frame_adv.size(0)
    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(1):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # ori audio
        filename_oriwav = os.path.join(prefix, 'audio_ori.wav')
        wavfile.write(os.path.join(args.vis, filename_oriwav), args.audRate, audio_ori[j].cpu().detach().numpy()[0])
        row_elements += [{'text': prefix}, {'audio': filename_oriwav}]

        # adv audio
        filename_advwav = os.path.join(prefix, 'audio_adv.wav')
        wavfile.write(os.path.join(args.vis, filename_advwav), args.audRate, audio_adv[j].detach().cpu().numpy()[0])

        # output images
        filename_oriimage = os.path.join(prefix, 'frame_ori.jpg')
        filename_advimage = os.path.join(prefix, 'audio_adv.jpg')
        imsave(os.path.join(args.vis, filename_oriimage), (255*frame_ori[j, :, :, :].cpu().detach().numpy().transpose((1, 2, 0))).astype(np.uint8))
        imsave(os.path.join(args.vis, filename_advimage),  (255*frame_adv[j, :, :, :].cpu().detach().numpy().transpose((1, 2, 0))).astype(np.uint8))
        row_elements += [
            {'audio':filename_advwav},
            {'image': filename_oriimage},
            {'image': filename_advimage}]

        vis_rows.append(row_elements)

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    feats_a = []
    feats_v = []
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']
        audio = audios[0].to(args.device)
        frame = frames[0].to(args.device).squeeze(2)
        gt = gts[0].to(args.device)


        # forward pass
        netWrapper.zero_grad()
        output, feat_v, feat_a = netWrapper.forward(frame, audio)
        feats_v.append(feat_v.detach())
        feats_a.append(feat_a.detach())
        err = criterion(output, gt) + F.cosine_similarity(feat_v, feat_a, 1).mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_classifier: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_classifier,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())



def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_classifier) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_classifier.state_dict(),
               '{}/classifier_{}'.format(args.ckpt, suffix_latest))

    cur_acc = history['val']['acc'][-1]
    if cur_acc > args.best_acc:
        args.best_acc = cur_acc
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_classifier.state_dict(),
                   '{}/classifier_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_frame, net_classifier) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_classifier.parameters(), 'lr': args.lr_classifier},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_classifier *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier)
    nets = (net_sound, net_frame, net_classifier)
    # crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    print(args.weights_Dv)
    args.Dv = torch.load(
        os.path.join(args.weights_Dv, 'Dv_best.pth')).cuda().detach()[:, :1024]
    args.Da = torch.load(
        os.path.join(args.weights_Da, 'Da_best.pth')).cuda().detach()[:, :1024]
    args.I = torch.eye(args.Da.size(1))
    # Wrap networks
    netWrapper = NetWrapper(args, nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)


    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'acc':[], 'cos':[]}}

    # Eval mode
    evaluate_adv(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):

        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')




if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_classifier)
        args.id += '-{}'.format(args.dic)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.vis, remove=True)
    elif args.mode == 'eval':
        makedirs(args.vis, remove=True)
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
        args.weights_classifier = os.path.join(args.ckpt, 'classifier_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_acc = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
