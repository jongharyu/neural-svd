from methods.nestedlora import NestedLoRA
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
    elif method_name == 'neuralsvd':
        method = NestedLoRA(
            model=model,
            neigs=args.neigs,
            step=args.loss.neuralsvd.step,
            sort=args.sort,
            sequential=args.loss.neuralsvd.sequential,
        )
    else:
        raise NotImplementedError
    return method
