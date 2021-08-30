import pickle
import sys
import datetime
import argparse
from torch import optim
from torch.nn.utils import clip_grad_norm_

sys.path.append(".")

import utils
from utils import *
from model_encoder import ModelForTraining


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

path_data_root = '../lpd_dataset/'
path_train_data = os.path.join(path_data_root, 'lpd_5_ccdepr_mix_v4_10000.npz')


def train_dp():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-l', '--lr', default=0.0001)
    parser.add_argument('-b', '--batch_size', default=4)
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    init_lr = float(args.lr)
    batch_size = int(args.batch_size)
    DEBUG = args.name == "debug"

    params = {
        "DECAY_EPOCH"   : [],
        "DECAY_RATIO"   : 0.1,
    }

    log("name:", args.name)
    log("args", args)

    if DEBUG:
        log("DEBUG MODE")
    else:
        utils.flog = open("../logs/" + args.name + ".log", "w")

    # hyper params
    n_epoch = 4000
    max_grad_norm = 3


    # config
    train_data = np.load(path_train_data)
    train_x = train_data['x'][:, :, [1, 0, 2, 3, 4, 5, 6]]
    train_y = train_data['y'][:, :, [1, 0, 2, 3, 4, 5, 6]]
    train_mask = train_data['decoder_mask'][:, :9999]
    num_batch = len(train_x) // batch_size
    
    # create saver
    saver_agent = Saver(exp_dir="../exp/" + args.name, debug=DEBUG)

    decoder_n_class = np.max(train_x, axis=(0, 1)) + 1
    # log
    log('num of encoder classes:', decoder_n_class)

    # init

    net = torch.nn.DataParallel(ModelForTraining(decoder_n_class))

    if torch.cuda.is_available():
        net.cuda()

    DEVICE_COUNT = torch.cuda.device_count()
    log("DEVICE COUNT:", DEVICE_COUNT)
    log("VISIBLE: " + os.environ["CUDA_VISIBLE_DEVICES"])

    net.train()
    n_parameters = network_paras(net)
    log('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    if args.path is not None:
        print('[*] load model from:', args.path)
        net.load_state_dict(torch.load(args.path))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    log('    train_data:', path_train_data.split("/")[-2])
    log('    num_batch:', num_batch)
    log('    train_x:', train_x.shape)
    log('    train_y:', train_y.shape)
    log('    train_mask:', train_mask.shape)
    log('    lr_init:', init_lr)
    for k, v in params.items():
        log(f'    {k}: {v}')

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(7)

        if epoch in params['DECAY_EPOCH']:
            log('LR decay by ratio', params['DECAY_RATIO'])
            for p in optimizer.param_groups:
                p['lr'] *= params['DECAY_RATIO']

        for bidx in range(num_batch):  # num_batch
            saver_agent.global_step_increment()

            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).long()
            batch_y = torch.from_numpy(batch_y).long()
            batch_mask = torch.from_numpy(batch_mask).float()

            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_mask = batch_mask.cuda()

            # run
            losses = net(batch_x, batch_y, batch_mask)
            losses = [l.sum()/DEVICE_COUNT for l in losses]
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7

            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Loss: {:.3f} | barbeat {:.3f}, type {:.3f}, pitch {:.3f}, duration {:.3f}, instr {:.3f}, o_den {:.3f}, b_den {:.3f}\r'.format(
                bidx, num_batch, float(loss), losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        log('-' * 80)
        log(time.ctime() + ' epoch: {}/{} | Loss: {:.3f} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        each_loss_str = 'barbeat {:.3f}, type {:.3f}, pitch {:.3f}, duration {:.3f}, instr {:.3f}, o_den {:.3f}, b_den {:.3f}\r'.format(
            acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5], acc_losses[6])
        log('each loss > ' + each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy
        loss = epoch_loss
        if 0.2 < loss <= 0.5:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif 0.001 < loss <= 0.20:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif loss <= 0.001:
            log('Finished')
            return
        else:
            saver_agent.save_model(net, name='loss_high')


if __name__ == '__main__':
    train_dp()
