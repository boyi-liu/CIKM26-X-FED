import importlib

model_params = {
    'cifar10': {
    },
    'cifar100': {
    },
    'tinyimagenet': {}
}

def match_dataset(dataset_arg):
    if 'cifar10' in dataset_arg and 'cifar100' not in dataset_arg:
        return 10
    elif 'cifar100' in dataset_arg:
        return 100
    elif 'tinyimagenet' in dataset_arg:
        return 200
    elif 'agnews' in dataset_arg:
        return 4
    else:
        return -1


def load_model(args):
    dataset_arg = args.dataset
    args.class_num = match_dataset(dataset_arg)
    if args.class_num == -1:
        exit('Dataset params not exist (in config.py)!')

    model_arg = args.model
    params = None

    if dataset_arg in model_params.keys() and model_arg in model_params[dataset_arg].keys():
        params = {**model_params[dataset_arg][model_arg]}

    model_module = importlib.import_module(f'models.{model_arg}')
    return getattr(model_module, model_arg)(args, params)
