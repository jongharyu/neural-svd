from methods.nestedlora import NestedLoRAForCDK


def get_cdk_method(args, model):
    if args.loss.name == 'neuralsvd':
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
