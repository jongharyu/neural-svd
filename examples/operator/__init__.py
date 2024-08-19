import logging
import os
import time
from contextlib import suppress

import numpy as np
import pandas as pd
import torch
from torch_ema import ExponentialMovingAverage
from tqdm import trange

from examples.operator.pde.ewm import EWMMonitor
from examples.operator.pde.plot import plot_1d_eigfuncs, plot_2d_eigfuncs
from examples.utils import get_optimizer
from methods.spectrum import compute_spectrum_evd, plot_and_save_spectrum

log = logging.getLogger(__name__)


def train_operator(
        args,
        method,
        operator,
        make_batch_ftn_train,
        val_data,
        batch_ftn_val,
        log_writer,
        log_file,
        device,
        importance_train,
        importance_val,
        ground_truth_spectrum=None,
):
    optimizer = get_optimizer(args, method)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_iters)
    ema = ExponentialMovingAverage(method.parameters(), decay=args.ema_decay)
    amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    all_eigvals = []
    all_norms = []
    last_log = 0.
    blowup_threshold = 0.5
    monitors_quadform = [EWMMonitor(blowup_thre=blowup_threshold) for i in range(args.neigs)]
    monitors_sqnorm = [EWMMonitor(blowup_thre=blowup_threshold) for i in range(args.neigs)]
    start = time.time()
    total_loss = 0.
    steps = trange(
        0,
        args.num_iters,
        initial=0,
        total=args.num_iters,
        desc='training',
        disable=None,
    )
    for it in steps:
        method.train()
        optimizer.zero_grad()
        # draw samples
        x = make_batch_ftn_train().to(device)
        x = x.reshape(x.shape[0], -1)
        with amp_autocast():
            loss, aux_outputs = method.compute_loss_operator(
                operator,
                x,
                importance=importance_train,
            )
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args.use_lr_scheduler and not (scale > scaler.get_scale()):
            scheduler.step()
        ema.update()  # Update the moving average with the new parameters from the last optimizer step
        total_loss += loss.item()
        if args.print_local_energies:
            f, Tf = aux_outputs['f'], aux_outputs['Tf']
            quadforms = (f * Tf).detach().cpu().numpy()  # (B, L)
            sqnorms = (f ** 2).detach().cpu().numpy()  # (B, L)
            for i, (monitor_quadform, monitor_sqnorm) in enumerate(zip(monitors_quadform, monitors_sqnorm)):
                monitor_quadform.update(quadforms[:, i])
                monitor_sqnorm.update(sqnorms[:, i])
            sqnorms_mean = [monitor.mean_of('mean_slow') for monitor in monitors_sqnorm]
            # By default, we print out the online estimates of squared norms
            energies_rep = '|'.join([f'{energy:S}' for energy in sqnorms_mean])
            steps.set_postfix(E=f'{energies_rep}')
            now = time.time()
            if (now - last_log) > 60:
                log.info(f'Progress: {it + 1}/{args.num_iters}, energy = {energies_rep}')
                last_log = now
            if (it + 1) % args.print_freq == 0:
                quads = [monitor.mean_of('mean_slow') for monitor in monitors_quadform]
                print(pd.DataFrame(data=np.stack([sqnorms_mean, np.array(quads) / np.array(sqnorms_mean)], axis=1),  # values
                                   index=np.arange(1, args.neigs + 1),
                                   columns=["Norms^2", "Rayleigh"]))
        if (it + 1) % args.print_freq == 0:
            write_dict = {
                'iter': it + 1,
                'train_loss': loss.item(),
                'avg_train_loss': total_loss / (it + 1),
                'time': time.time() - start,
            }
            print(write_dict)
            print(f'learning rate is {scheduler.get_last_lr()}')
            log_writer.writerow(write_dict)
            log_file.flush()
        if (it + 1) % args.eval_freq == 0:
            method.eval()
            with ema.average_parameters():
                if batch_ftn_val is not None:
                    normalize = method.name in ['nestedlora', 'neuralsvd']
                    outputs = compute_spectrum_evd(
                        method,
                        dataloader=batch_ftn_val(),
                        operator=operator,
                        importance_train=importance_train,
                        importance_val=importance_val,
                        post_align=args.post_align,
                        normalize=normalize,
                        set_first_mode_const=False,
                        device=device,
                    )
                    plot_and_save_spectrum(
                        {'RQ': outputs['eigvals'], 'Norms^2': outputs['norms'] if normalize else None},
                        outputs['cov'],
                        ground_truth_spectrum=ground_truth_spectrum,
                        ylim=None,
                        log_dir=args.log_dir,
                        tag=f'it{it + 1}',
                        termplot=True
                    )
                    print(f"it{it + 1} eigvals: {outputs['eigvals']}")
                    print(f"it{it + 1} norms: {outputs['norms']}")
                    all_eigvals.append(outputs['eigvals'])
                    all_norms.append(outputs['norms'])
            if args.ndim == 1:
                plot_1d_eigfuncs(val_data.cpu().numpy(), outputs['eigfuncs'], log_dir=args.log_dir, tag=f'it{it + 1}')
            if args.ndim == 2:
                plot_2d_eigfuncs(outputs['eigfuncs'], log_dir=args.log_dir, tag=f'it{it + 1}')
            model_path = os.path.join(args.log_dir, f"{it + 1}.pth")
            ckpt = dict(args=args,
                        method=method.state_dict(),
                        ema=ema.state_dict(),
                        optimizer=optimizer.state_dict())
            torch.save(ckpt, model_path)
            print(f"Model saved to {model_path}")
            if method.name == 'spinx':
                method.update_weights_operator(
                    operator,
                    x,
                    importance_train,
                    args.split_batch,
                )
    return all_eigvals, all_norms
