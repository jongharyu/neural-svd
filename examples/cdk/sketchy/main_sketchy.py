import gc
import json
import os
from functools import partial

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from examples import opts
from examples.cdk.sketchy.data import SketchyVGGDataLoader
from examples.cdk.sketchy.retrieve import SketchyRetrieval
from examples.cdk.optimizers import get_optimizer_with_params
from examples.cdk.utils import plot_hist_ratios_wrapper
from examples.models.mlp import get_mlp
from examples.models.siam import HeteroNetwork
from examples.opts import parse_loss_configs
from examples.utils import get_log_file
from methods.cdk import get_cdk_method
from methods.utils import parse_str
from methods.spectrum import compute_spectrum_svd, plot_and_save_spectrum
from tools.generic import set_deterministic

plt.style.use('ggplot')
IMPLEMENTED_LOSSES = ('neuralsvd', 'neuralef', )


def get_args():
    parser = configargparse.ArgumentParser(description='Sketch-based photo retrieval (Sketchy dataset)')
    opts.amp_opts(parser)
    opts.reg_opts(parser)
    opts.loss_opts(parser)

    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--exp_tag', type=str)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--overwrite', action='store_true', default=False)

    # Sketchy specific
    parser.add_argument('--root_dir', default='~')
    parser.add_argument('--sketchy_split', type=str, default=1)
    parser.add_argument('--sketchy_retrieval_metric', type=str, default='inner_product',
                        choices=['euclidean', 'cosine', 'inner_product'])
    parser.add_argument('--save_retrieved_images', action='store_true', default=False)
    parser.add_argument('--n_retrievals_to_save', type=int, default=10)
    parser.add_argument('--n_retrievals', type=int, default=100)
    parser.add_argument('--ap_ver', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--trunc_dims', nargs='*', type=int, default=[])
    parser.add_argument('--randperm', action='store_true')

    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--return_map_all', action='store_true')
    parser.add_argument('--log_freq', type=int, default=1)

    # optimizer
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--momentum', default=0., type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--base_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=0., type=float)
    parser.add_argument('--warmup_lr', default=0., type=float)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False)

    # model config
    parser.add_argument('--network_dims', type=str, default='512,512')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--activation', type=str, default='lrelu0.2')

    args = parser.parse_args()
    args = parse_loss_configs(args, IMPLEMENTED_LOSSES)

    return args


def main(args):
    # set random seed
    gpu = 0
    set_deterministic(args.seed)

    log_file, log_writer = get_log_file(args.log_dir,
                                        fieldnames=['epoch', 'lr', 'loss',
                                                    'test_loss', 'test_P@K', 'test_mAP@all',
                                                    'valid_P@K', 'valid_mAP@all'])

    # load data
    train_loader = SketchyVGGDataLoader(args.batch_size, shuffle=True, drop_last=False,
                                        root_path=args.root_dir, split=args.sketchy_split,
                                        train_or_test='train')
    test_loader = SketchyVGGDataLoader(args.batch_size, shuffle=False, drop_last=False,
                                       root_path=args.root_dir, split=args.sketchy_split,
                                       train_or_test='test')
    valid_loader = SketchyVGGDataLoader(args.batch_size, shuffle=False, drop_last=False,
                                        root_path=args.root_dir, split=args.sketchy_split,
                                        train_or_test='valid')

    # define models
    input_dim = 512  # vgg feature dimension (both photo and sketch)
    net_sizes = [input_dim] + parse_str(args.network_dims)
    model = HeteroNetwork(backbones=[get_mlp(sizes=net_sizes,
                                             bias=True, nonlinearity=args.activation, use_bn=args.use_bn),
                                     get_mlp(sizes=net_sizes,
                                             bias=True, nonlinearity=args.activation, use_bn=args.use_bn)],
                          projectors=[nn.Identity(), nn.Identity()],
                          mu=args.mu,
                          regularize_mode=args.regularize_mode).to(gpu)
    args.output_dim = model.output_dims['x']

    sketchy_retrieval = SketchyRetrieval(test_loader,
                                         n_images_to_save=args.n_retrievals_to_save,
                                         n_retrievals=args.n_retrievals,
                                         metric=args.sketchy_retrieval_metric,
                                         run_path=args.log_dir,
                                         device=gpu)
    sketchy_retrieval_valid = SketchyRetrieval(valid_loader,
                                               n_images_to_save=args.n_retrievals_to_save,
                                               n_retrievals=args.n_retrievals,
                                               metric=args.sketchy_retrieval_metric,
                                               run_path=args.log_dir,
                                               device=gpu)

    # define loss function
    method = get_cdk_method(args, model)

    # for NeuralSVD, we check and plot spectrum
    spectrum_dir = None
    if args.loss.name in ['neuralsvd', 'neuralef']:
        spectrum_dir = os.path.join(args.log_dir, "spectrum")
        if not os.path.exists(spectrum_dir):
            os.makedirs(spectrum_dir)

    # define optimizer
    optimizer = get_optimizer_with_params(
        args.optimizer, model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # lr_scheduler = LRScheduler(
    #     optimizer,
    #     args.warmup_epochs,
    #     args.warmup_lr * args.batch_size / 256,
    #     args.num_epochs,
    #     args.base_lr * args.batch_size / 256,
    #     args.final_lr * args.batch_size / 256,
    #     train_loader.max_steps,
    #     constant_predictor_lr=True  # See the end of Section 4.2
    # )

    # gradient scaler for mixed precision
    # reference: https://github.com/pytorch/pytorch/issues/40497#issuecomment-1262373602
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs * train_loader.max_steps)

    save_path = f"{args.log_dir}/best_stats_{','.join([str(s) for s in sorted(args.trunc_dims)])}.npz"
    if os.path.exists(save_path):
        print(f'The current experiment has been done already!')
        return

    if not args.eval_only:
        best_prec_at_K, best_map = np.array(0.), np.array(0.)
        for epoch in range(args.num_epochs):
            model.train()
            train_cnt = 0
            train_cum_loss = 0.
            skipped_updates = 0  # we can count the number of updates with nan grads for debugging
            print(f"Epoch {epoch + 1}/{args.num_epochs} (#iters={train_loader.max_steps})")

            rs_pxy_train, rs_pxpy_train = [], []
            correction_cnts = dict(none=0, upper=0, lower=0, both=0)
            for idx, (x, y, cls) in enumerate(train_loader):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=not args.disable_amp):
                    # TODO: plug-in joint_mask
                    # cls = cls.unsqueeze(1).cuda(gpu, non_blocking=True)  # (B, 1)
                    # joint_mask = (cls == cls.T)  # (B, B)
                    _, fx_emb, _, fy_emb = method(
                        x.cuda(gpu, non_blocking=True),
                        y.cuda(gpu, non_blocking=True)
                    )
                    loss, *_, rs_joint, rs_indep = method.compute_loss(fx_emb, fy_emb)
                    rs_pxy_train.append(rs_joint.detach().cpu())
                    rs_pxpy_train.append(rs_indep.detach().cpu())

                scaler.scale(loss).backward()
                if args.clip_grad_norm:
                    scaler.unscale_(optimizer)
                    if args.optimizer != 'lars':
                        total_norm = nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=args.clip_grad_norm_value)  # grad clip helps in both amp and fp32
                        if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
                            # scaler is going to skip optimizer.step() if grads are nan or inf
                            # some updates are skipped anyway in the amp mode, but we can count for statistics
                            skipped_updates += 1
                            print(f"\tWarning (gpu={gpu}): gradient norm is {total_norm} with loss={loss} "
                                  f"{f'(scaler._scale={scaler._scale.item()})!' if not args.disable_amp else ''}")
                scaler.step(optimizer)
                scaler.update()
                if args.use_lr_scheduler:
                    lr_scheduler.step()
                train_cum_loss += loss.sum().detach().cpu().numpy()
                train_cnt += 1

            print(f"In Epoch {epoch + 1}: \n"
                  f"\t - the number of skipped updates is {skipped_updates}\n"
                  f"\t - the number of corrections are {correction_cnts}")

            model.eval()
            test_cnt = 0
            test_cum_loss = 0.
            with torch.no_grad():
                rs_pxy_test, rs_pxpy_test = [], []
                for idx, (x, y, cls) in enumerate(tqdm(test_loader)):
                    with torch.cuda.amp.autocast(enabled=not args.disable_amp):
                        _, fx_emb, _, fy_emb = method(
                            x.cuda(gpu, non_blocking=True),
                            y.cuda(gpu, non_blocking=True)
                        )
                        # TODO: plug-in joint_mask
                        # cls = cls.unsqueeze(1).cuda(gpu, non_blocking=True)  # (B, 1)
                        # joint_mask = (cls == cls.T)  # (B, B)
                        loss_nested, *_, rs_pxy_, rs_pxpy_ = method.compute_loss(fx_emb, fy_emb)
                        test_loss = loss_nested  # will log the nested objective
                        rs_pxy_test.append(rs_pxy_.detach().cpu())
                        rs_pxpy_test.append(rs_pxpy_.detach().cpu())

                    test_cum_loss += test_loss.sum().detach().cpu().numpy()
                    test_cnt += 1

            print(f"Retrieve w.r.t. test set...")
            precision_Ks, average_precisions = sketchy_retrieval.evaluate(
                lambda x: partial(model.forward_single, x_or_y='x')(x)[1],
                lambda x: partial(model.forward_single, x_or_y='y')(x)[1],
                epoch=epoch + 1,
                save_retrieved_images=args.save_retrieved_images,
                return_map_all=args.return_map_all,
                ap_ver=args.ap_ver,
                tag='test',
            )

            precision_Ks_valid, average_precisions_valid = np.array(0.), np.array(0.)
            if len(valid_loader) > 0:
                print(f"Retrieve w.r.t. valid set...")
                precision_Ks_valid, average_precisions_valid = sketchy_retrieval_valid.evaluate(
                    lambda x: partial(model.forward_single, x_or_y='x')(x)[1],
                    lambda x: partial(model.forward_single, x_or_y='y')(x)[1],
                    epoch=epoch + 1,
                    save_retrieved_images=args.save_retrieved_images,
                    return_map_all=args.return_map_all,
                    ap_ver=args.ap_ver,
                    tag=f'valid',
                )
            if best_prec_at_K <= precision_Ks_valid.mean():
                best_prec_at_K = precision_Ks_valid.mean()
                model_path = os.path.join(args.ckpt_dir, f"best.pth")
                ckpt = dict(args=args,
                            epoch=epoch + 1,
                            model=model.state_dict()
                            )
                torch.save(ckpt, model_path)
                del ckpt
                print(f"Model saved to {model_path}")

            write_dict = {
                'epoch': epoch + 1,
                'lr': lr_scheduler.get_last_lr(),
                'loss': train_cum_loss / train_cnt,
                'test_loss': test_cum_loss / test_cnt,
                'test_P@K': precision_Ks.mean(),
                'test_mAP@all': average_precisions.mean(),
                'valid_P@K': precision_Ks_valid.mean(),
                'valid_mAP@all': average_precisions_valid.mean(),
            }
            print(json.dumps(write_dict, indent=4))
            log_writer.writerow(write_dict)
            log_file.flush()

            # save checkpoint for possible breakdown
            ckpt = dict(args=args,
                        epoch=epoch + 1,
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict())
            torch.save(ckpt, os.path.join(args.ckpt_dir, 'ckpt.pth'))
            del ckpt
            gc.collect()

            if (epoch + 1) % args.log_freq == 0:
                if args.loss.name in ['neuralsvd', 'neuralef']:
                    print(f"Checking orthonormality after epoch {epoch + 1}...")
                    def extract_emb(x, y):
                        outputs = method(x, y)
                        return outputs[1], outputs[3]
                    spectrum, orthogonality_x, orthogonality_y = compute_spectrum_svd(
                        extract_emb,
                        test_loader,
                        gpu=gpu,
                        set_first_mode_const=True
                    )
                    plot_and_save_spectrum(
                        {'singvals': spectrum},
                        orthogonality_x,
                        orthogonality_y,
                        log_dir=spectrum_dir,
                        tag=f'test({epoch + 1})'
                    )

                if args.loss.name == 'neuralsvd':
                    plot_hist_ratios_wrapper(rs_pxy_train, rs_pxpy_train, rs_pxy_test, rs_pxpy_test, tag=f'ep={epoch + 1}')

    ckpt = torch.load(os.path.join(args.ckpt_dir, f"best.pth"), map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    perm = torch.arange(net_sizes[-1])
    if args.randperm:
        assert net_sizes[-1] == args.loss.neuralsvd.step, "Apply random permutation only when step=feature_dim"
        perm = torch.randperm(net_sizes[-1])

    prec_at_Ks, map_at_alls = [], []
    with torch.no_grad():
        for trunc_dim in sorted(args.trunc_dims):
            print(f"Retrieve using {trunc_dim}-dims w.r.t. test set...")
            if trunc_dim > 0:
                forward_x = lambda x: partial(model.forward_single, x_or_y='x')(x)[1][:, perm[:trunc_dim]]
                forward_y = lambda y: partial(model.forward_single, x_or_y='y')(y)[1][:, perm[:trunc_dim]]
            else:
                forward_x = lambda x: partial(model.forward_single, x_or_y='x')(x)[1][:, perm[trunc_dim:]]
                forward_y = lambda y: partial(model.forward_single, x_or_y='y')(y)[1][:, perm[trunc_dim:]]
            precision_Ks, average_precisions = sketchy_retrieval.evaluate(
                forward_x,
                forward_y,
                epoch=0,
                save_retrieved_images=True,
                return_map_all=True,
                ap_ver=args.ap_ver,
                tag=f'test{trunc_dim}',
            )
            prec_at_Ks.append(precision_Ks.mean())
            map_at_alls.append(average_precisions.mean())

        prec_at_Ks = np.array(prec_at_Ks)
        map_at_alls = np.array(map_at_alls)

    np.savez(save_path,
             trunc_dims=sorted(args.trunc_dims),
             prec_at_Ks=prec_at_Ks,
             map_at_alls=map_at_alls)


if __name__ == "__main__":
    args = get_args()

    # parse loss configs
    args.log_dir = os.path.join(
        args.log_dir,
        'sketchy',
        f'split{args.sketchy_split}',
        f'{args.loss.name}_{"seq" if args.loss.neuralsvd.sequential else f"jnt_step{args.loss.neuralsvd.step}"}_mu={args.mu}'
        f'_mlp{args.network_dims}'
        f'{"_sch" if args.use_lr_scheduler else ""}',
        f'{args.exp_tag}_{args.optimizer}lr{args.base_lr}_bs{args.batch_size}e{args.num_epochs}_seed{args.seed}'
    )

    if os.path.exists(args.log_dir) and not args.overwrite:
        raise ValueError(f"{args.log_dir} already exists and overwrite is not permitted")
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # for NeuralSVD, we check and plot spectrum
    args.spectrum_dir = None
    if args.loss.name in ['neuralsvd', 'neuralef']:
        args.spectrum_dir = os.path.join(args.log_dir, "spectrum")
        if not os.path.exists(args.spectrum_dir):
            os.makedirs(args.spectrum_dir)

    print(args)
    main(args)
