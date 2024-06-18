

from anchor_based.dsnet import DSNet, DSNet_DeepAttention, DSNet_MultiAttention
from anchor_based.dsnet import DSNetTriangularAttention, DSNetMotionFeatures
from anchor_free.dsnet_af import DSNetAF, DSNetAF_DeepAttention
from helpers import init_helper
from anchor_based.fourier_ab_models import DSNet_fourier


def get_anchor_based( base_model, num_feature, num_hidden, anchor_scales,
                     num_head, fc_depth, attention_depth, encoder_type, **kwargs):
    args = init_helper.get_arguments()
    if args.model_depth == 'shallow':
        return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head, fc_depth)
    elif args.model_depth == 'deep':
        return DSNet_DeepAttention(base_model, num_feature, num_hidden, anchor_scales, num_head, fc_depth, attention_depth)
    elif args.model_depth == 'local-global-attention':
        return DSNet_MultiAttention(base_model, num_feature, num_hidden, anchor_scales, num_head)
    elif args.model_depth == 'triangular':
        return DSNetTriangularAttention(base_model, num_feature, num_hidden, anchor_scales, num_head)
    elif args.model_depth == 'cross-attention':
        return DSNetMotionFeatures(base_model, num_feature, num_hidden, anchor_scales, num_head, attention_depth, encoder_type)
    

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
