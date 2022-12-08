import numpy as np
import cupy as cp
import chainer
from models.model import ElmanId
from utils.graphics import data2img
from utils.functions import boolean

import argparse
import wandb
import matplotlib.pyplot as plt
from tqdm import trange


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',         default=1,       type=int)
    parser.add_argument('--idx-size',    default=100,     type=int)
    parser.add_argument('--N',           default=200,     type=int)
    parser.add_argument('--in-size',     default=100,     type=int)
    parser.add_argument('--norm',        default=False,   type=boolean)
    parser.add_argument('--p',           default=None,    type=int)
    parser.add_argument('--batch',       default=100,     type=int)
    parser.add_argument('--epoch',       default=10,      type=int)
    parser.add_argument('--episode',     default=1000,     type=int)
    parser.add_argument('--T',           default=1000,    type=int)
    parser.add_argument('--bptt',        default=50,      type=int)
    parser.add_argument('--model-type',  default='elman', type=str, choices=('elman', 'zhang'))
    parser.add_argument('--coef-neuron', default=0.0,     type=float)
    parser.add_argument('--coef-rnn',    default=0.0,     type=float)
    parser.add_argument('--reg',         default='l2',    type=str, choices=('l2', 'l1'))
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # data
    data = np.load('./data/dim={}_False_B={}_T={}_N={}.npz'.format(args.dim, args.batch, args.T, args.N))
    rs = cp.asarray(data['rs'])
    xs = cp.asarray(data['xs'])
    x0 = cp.asarray(data['x0'])
    vs = cp.asarray(data['vs'])

    # name
    model_type = args.model_type + '-' + str(args.p) if args.p is not None else args.model_type
    name = '{}_{}_{}of{}-{}'.format(args.dim, model_type, args.idx_size, args.N, args.in_size)
    wandb.init(project="nsi", entity="liuyuezhang", name=name, config=args)

    # idx
    idx = np.arange(args.idx_size)
    data_idx = idx
    np.savez(wandb.run.dir + '/idx', data_idx=data_idx, idx=idx)

    # model
    model = ElmanId(N=args.N, in_size=args.in_size, idx=idx, feature_size=args.dim, nonlinear='relu-tanh', alpha=0.2, norm=args.norm, bias=True,
                    noise_scale=0.0, lr=1e-4, reg=args.reg, coef_neuron=args.coef_neuron, coef_rnn=args.coef_rnn).to_gpu()

    # train loop
    chainer.serializers.save_npz(wandb.run.dir + '/model-{}.pkl'.format(0), model)
    W = cp.asnumpy(model.W.array)
    img_W = data2img(W)
    plt.imsave(wandb.run.dir + '/{}.png'.format(0), img_W)

    step = 0
    for e in trange(1, args.epoch + 1):
        for _ in trange(args.episode):
            t0 = np.random.randint(0, args.T - args.bptt)
            model.init(rs[:, t0, data_idx])
            for t in range(1, args.bptt):
                model.step(v=vs[:, t0 + t, :], x=rs[:, t0 + t, data_idx])
                step += args.batch
            # update
            loss_total, loss, loss_neuron, loss_rnn = model.update()
            # log
            wandb.log({"step": step, "loss_total": loss_total, "loss": loss, "loss_neuron": loss_neuron, "loss_rnn": loss_rnn})

        # save
        chainer.serializers.save_npz(wandb.run.dir + '/model-{}.pkl'.format(e), model)
        if args.p is None:
            W = cp.asnumpy(model.W.array)
        else:
            W = cp.asnumpy(model.W1.array @ model.W2.array.T)
        img_W = data2img(W)
        plt.imsave(wandb.run.dir + '/{}.png'.format(e), img_W)


if __name__ == '__main__':
    main()
