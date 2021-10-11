from src.models.backbones.utils import is_model, is_model_in_modules, model_entrypoint, list_models, load_checkpoint


def create_backbone(
        model_name,
        pretrained=False,
        in_chans=3,
        checkpoint_path='',
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """

    margs = dict(pretrained=pretrained, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)

        pretrained_models = list_models(pretrained=True)
        pretrain = pretrained and (model_name in pretrained_models)
        margs['pretrained'] = pretrain
        if pretrained and not pretrain:
            print("WARNING: the model doesn't have pretrained weights")
        elif pretrain:
            print(f'Pretrained weights for {model_name} will be downloaded')

        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
