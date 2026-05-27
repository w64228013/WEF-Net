import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import MRViewDataset_text_combined
from loss_function import base_edl_loss
from model_zcy_v4 import SwinEDL_03_clinic
import time

from log_util import generate_logger

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from warnings import filterwarnings

filterwarnings('ignore')

np.set_printoptions(precision=4, suppress=True)


class cls_eval():
    def __init__(self, n_class=2):
        self.n_class = n_class

    def __call__(self, pred, truth):
        auc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        acc_list = []

        pred = np.argmax(pred, axis=1)

        temp_auc = roc_auc_score(truth, pred)

        temp_acc = accuracy_score(truth, pred)
        temp_prec = precision_score(truth, pred)
        temp_rec = recall_score(truth, pred)
        temp_f1 = f1_score(truth, pred)

        auc_list.append(temp_auc)
        acc_list.append(temp_acc)
        prec_list.append(temp_prec)
        rec_list.append(temp_rec)
        f1_list.append(temp_f1)

        return auc_list, acc_list, prec_list, rec_list, f1_list


def save_model(args, model, epoch, metric=None):
    save_path = args.save_weight_path
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    if metric != None:
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_{metric:.6f}.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_final.pth'))


def swin_edl_fuse_text(args):
    logger = generate_logger(os.path.join(args.save_weight_path, 'log.txt'))

    num_classes = 2
    data_src_path = os.environ.get('MR_DATA_ROOT', r'/public/home/b_tyxia/data/250819_MR/')
    train_loader = DataLoader(
        MRViewDataset_text_combined(data_files=args.train_file, data_src_path=data_src_path, text_file=args.train_text),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(
        MRViewDataset_text_combined(data_files=args.test_file, data_src_path=data_src_path, text_file=args.test_text),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = SwinEDL_03_clinic(num_classes=num_classes)
    model = model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-6)

    best_metric = -1
    gamma = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f'====> {epoch}')
        sampleNum = 0
        for batch_idx, (dwi_t1_t2_data, dwi_t1_t2_WMH_data, clinic_data, target) in enumerate(train_loader):

            dwi_t1_t2_data = dwi_t1_t2_data.cuda()
            dwi_t1_t2_WMH_data = dwi_t1_t2_WMH_data.cuda()
            clinic_data = clinic_data.cuda()
            target = target.cuda()

            evidence_list = model(dwi_t1_t2_data, dwi_t1_t2_WMH_data, clinic_data)
            loss1 = base_edl_loss(evidence_list[0], target, epoch, num_classes, args.annealing_step, gamma, device)
            loss2 = base_edl_loss(evidence_list[1], target, epoch, num_classes, args.annealing_step, gamma, device)
            loss3 = base_edl_loss(evidence_list[2], target, epoch, num_classes, args.annealing_step, gamma, device)
            loss4 = base_edl_loss(evidence_list[3], target, epoch, num_classes, args.annealing_step, gamma, device)
            loss = loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            sampleNum += len(dwi_t1_t2_data[0])

            if (batch_idx + 1) % int(5) == 0:
                outputString = 'Train Epoch:{} [{}/{}] loss:{:.6f} LR:{}' \
                    .format(epoch, sampleNum, len(train_loader.dataset), loss / sampleNum,
                            lr_scheduler.get_last_lr()[0])
                logger.info(outputString)

        ########################################################################
        model.eval()
        num_correct, num_sample = 0, 0
        pred_np = []
        truth_np = []

        for dwi_t1_t2_data, dwi_t1_t2_WMH_data, clinic_data, target in test_loader:
            dwi_t1_t2_data = dwi_t1_t2_data.cuda()
            dwi_t1_t2_WMH_data = dwi_t1_t2_WMH_data.cuda()
            clinic_data = clinic_data.cuda()
            target = target.cuda()

            with torch.no_grad():
                evidence_list = model(dwi_t1_t2_data, dwi_t1_t2_WMH_data, clinic_data)
                evidence = evidence_list[0]
                _, Y_pre = torch.max(evidence, dim=1)
                num_correct += (Y_pre == target).sum().item()
                num_sample += target.shape[0]

                pred_np.extend(list(evidence.cpu().detach().numpy()))
                truth_np.extend(list(target.cpu().detach().numpy()))

        pred_np = np.array(pred_np)
        truth_np = np.array(truth_np)
        auc_list, acc_list, prec_list, rec_list, f1_list = cls_eval(num_classes)(pred_np, truth_np)

        total_acc = num_correct / num_sample
        print('====> acc: {:.4f}, best_metric: {:.4f}'.format(total_acc, best_metric))
        print('====> auc_list:', auc_list)
        print('====> acc_list:', acc_list)
        print('====> prec_list:', prec_list)
        print('====> rec_list:', rec_list)
        print('====> f1_list:', f1_list)

        logger.info(
            f'epoch:{epoch}, best_metric:{best_metric}, acc:{total_acc}, auc_list:{auc_list}, acc_list:{acc_list}, prec_list:{prec_list}, rec_list:{rec_list}, f1_list:{f1_list}')

        if total_acc > best_metric:
            best_metric = total_acc
            save_model(args, model, epoch, metric=total_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate')
    parser.add_argument('--save-weight-path', type=str,
                        default=r'/public/home/b_tyxia/code/stroke_zcy/swinedl_z_test_JIANGDU_lr_1e2_batch_4_08_')

    parser.add_argument('--train-text', type=str, default=r'ZHONGDA_SUPP_train.txt')
    parser.add_argument('--test-text', type=str, default=r'ZHONGDA_SUPP_train.txt')
    parser.add_argument('--train-file', type=str, nargs='+', default=['ZHONGDA_SUPP_train.txt'])
    parser.add_argument('--test-file', type=str, nargs='+', default=[
        'ZHONGDA_SUPP_test.txt'])

    args = parser.parse_args()

    args.save_weight_path = args.save_weight_path + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    os.makedirs(args.save_weight_path, exist_ok=True)

    swin_edl_fuse_text(args)
