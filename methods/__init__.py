from methods.nestedlora import NestedLoRA, NestedLoRAForCDK
from methods.neuralef import NeuralEigenfunctions
from methods.spin import SpIN
from methods.spinx import SpINx


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
    elif method_name == 'neuralsvd':
        method = NestedLoRA(
            model,
            neigs=args.neigs,
            step=args.loss.neuralsvd.step,
            sort=args.sort,
            sequential=args.loss.neuralsvd.sequential,
            residual_weight=args.residual_weight,
        )
    else:
        raise NotImplementedError
    return method


def get_cdk_method(args, method_name, model):
    if method_name == 'neuralsvd':
        method = NestedLoRAForCDK(
            model,
            neigs=args.neigs,
            step=args.loss.neuralsvd.step,
            sequential=args.loss.neuralsvd.sequential,
            set_first_mode_const=args.loss.neuralsvd.set_first_mode_const,
        )
    else:
        raise NotImplementedError

    return method
