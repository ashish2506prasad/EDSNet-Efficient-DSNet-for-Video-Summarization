
from anchor_based.dsnet import DSNet, DSNet_DeepAttention, DSNet_MultiAttention
from anchor_based.dsnet import DSNetMotionFeatures
from anchor_free.dsnet_af import DSNetAF, DSNetAF_DeepAttention
from helpers import init_helper


def get_anchor_based( base_model, num_feature, num_hidden, anchor_scales,
                     num_head, fc_depth, attention_depth, encoder_type, orientation,**kwargs):
    args = init_helper.get_arguments()
    if args.model_depth == 'shallow':
        return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head, fc_depth, orientation)
    elif args.model_depth == 'deep':
        return DSNet_DeepAttention(base_model, num_feature, num_hidden, anchor_scales, num_head, fc_depth, attention_depth, orientation)
    elif args.model_depth == 'local-global-attention':
        return DSNet_MultiAttention(base_model, num_feature, num_hidden, anchor_scales, num_head, fc_depth, orientation)
    elif args.model_depth == 'cross-attention':
        return DSNetMotionFeatures(base_model, num_feature, num_hidden, anchor_scales, num_head, attention_depth, encoder_type)
    elif args.model_depth == 'original':
        return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head)
    

def get_anchor_free(base_model, num_feature, num_hidden, num_head, fc_depth, orientation, **kwargs):
    args = init_helper.get_arguments()
    if args.model_depth == 'shallow':
        return DSNetAF(base_model, num_feature, num_hidden, num_head, fc_depth, orientation)
    elif args.model_depth == 'deep':
        return DSNetAF_DeepAttention(base_model, num_feature, num_hidden, num_head, fc_depth, orientation)
    elif args.model_depth == 'original':
        return DSNetAF(base_model, num_feature, num_hidden, num_head, fc_depth, orientation)
    elif args.model_depth == 'local-global-attention':
        return DSNet_MultiAttention(base_model, num_feature, num_hidden, num_head)


def get_model(model_type, **kwargs):
    if model_type == 'anchor-based':
        return get_anchor_based(**kwargs)
    elif model_type == 'anchor-free':
        return get_anchor_free(**kwargs)
    else:
        raise ValueError(f'Invalid model type {model_type}')
