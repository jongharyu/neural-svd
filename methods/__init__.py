from methods.ssl.barlowtwins import BarlowTwinsLoss
from methods.neuralef import NeuralEigenfunctions, NeuralEigenmapsLossCDK, NeuralEigenmapsLoss
from methods.nestedlora import NestedLoRA, NestedLoRAForCDK
from methods.neuralsvd import NeuralSVDLoss, NeuralSVDLossFixedKernel, NeuralSVDLossCDK
from methods.spin import SpIN
from methods.spinx import SpINx
from methods.ssl.simclr import SimCLRLoss


def get_evd_method(args, method_name, model):
    if method_name == 'spin':
        method = SpIN(
            model=model,
            neigs=args.neigs,
            decay=args.loss.spin.decay,
            use_vmap=args.loss.spin.use_pfor,
        )
    elif method_name == 'spinx':
        method = SpINx(
            model=model,
            neigs=args.neigs,
            decay=args.loss.spin.decay,
        )
    elif method_name == 'neuralef':
        method = NeuralEigenfunctions(
            model=model,
            neigs=args.neigs,
            batchnorm_mode=args.loss.neuralef.batchnorm_mode,
            unbiased=args.loss.neuralef.unbiased,
            sort=args.sort,
        )
    elif method_name == 'neigenmaps':
        method = NeuralEigenmapsLoss(
            model=model,
            neigs=args.neigs,
            batchnorm_mode=args.loss.neuralef.batchnorm_mode,
            reg_weight=args.loss.neuralef.reg_weight
        )
    elif method_name == 'nestedlora':
        method = NestedLoRA(
            model,
            neigs=args.neigs,
            step=args.loss.neuralsvd.step,
            sort=args.sort,
            sequential=args.loss.neuralsvd.sequential,
            separation=args.loss.neuralsvd.separation,
            separation_mode=args.loss.neuralsvd.separation_mode,
            separation_init_scale=args.loss.neuralsvd.separation_init_scale,
            inner_product=args.loss.neuralsvd.inner_product,
            residual_weight=args.residual_weight,
        )
    elif method_name == 'neuralsvd':
        method = NeuralSVDLoss(
            model,
            step=args.loss.neuralsvd.step,
            neigs=args.neigs,
            sequential=args.loss.neuralsvd.sequential,
            separation=args.loss.neuralsvd.separation,
            separation_mode=args.loss.neuralsvd.separation_mode,
            separation_init_scale=args.loss.neuralsvd.separation_init_scale,
            stop_grad=args.loss.neuralsvd.stop_grad,
            along_batch_dim=args.loss.neuralsvd.along_batch_dim,
            weight_order=args.loss.neuralsvd.weight_order,
            oneshot=args.loss.neuralsvd.oneshot,
            unittest=args.loss.neuralsvd.unittest,
        )
    else:
        raise NotImplementedError
    return method


def get_cdk_method(args, method_name, model):
    if method_name == 'neuralsvd':
        loss_ftn = NestedLoRAForCDK(
            model,
            neigs=args.neigs,
            step=args.loss.neuralsvd.step,
            sequential=args.loss.neuralsvd.sequential,
            set_first_mode_const=args.loss.neuralsvd.set_first_mode_const,
            separation=args.loss.neuralsvd.separation,
            separation_mode=args.loss.neuralsvd.separation_mode,
            separation_init_scale=args.loss.neuralsvd.separation_init_scale,
        )
    elif method_name == 'barlowtwins':
        loss_ftn = BarlowTwinsLoss(
            feature_dim=args.model.output_dim,
            reg_weight=args.loss.barlowtwins.reg_weight,
            device=None)  # TODO: fix device?
    elif method_name == 'simclr':
        loss_ftn = SimCLRLoss(
            normalize=args.loss.simclr.normalize,
            temperature=args.loss.simclr.temperature
        )
    elif method_name == 'neuralef':
        loss_ftn = NeuralEigenmapsLossCDK(
            stop_grad=args.loss.neuralef.stop_grad,
            reg_weight=args.loss.neuralef.reg_weight
        )
    else:
        raise NotImplementedError

    return loss_ftn
