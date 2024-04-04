from anchor_based.dsnet import DSNet, DSNet_DeepAttention, DSNet_MultiAttention
from anchor_free.dsnet_af import DSNetAF, DSNetAF_DeepAttention
from helpers import init_helper, data_helper


def get_anchor_based(base_model, num_feature, num_hidden, anchor_scales,
                     num_head, **kwargs):
    args = init_helper.get_arguments()
    if args.model_depth == 'shallow':
        return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head)
    elif args.model_depth == 'deep':
        return DSNet_DeepAttention(base_model, num_feature, num_hidden, anchor_scales, num_head)
    elif args.model_depth == 'local-global-attention':
        return DSNet_MultiAttention(base_model, num_feature, num_hidden, anchor_scales, num_head)
    



def get_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    args = init_helper.get_arguments()
    if args.model_depth == 'shallow':
        return DSNetAF(base_model, num_feature, num_hidden, num_head)
    else:
        return DSNetAF_DeepAttention(base_model, num_feature, num_hidden, num_head)


def get_model(model_type, **kwargs):
    if model_type == 'anchor-based':
        return get_anchor_based(**kwargs)
    elif model_type == 'anchor-free':
        return get_anchor_free(**kwargs)
    else:
        raise ValueError(f'Invalid model type {model_type}')
