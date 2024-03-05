from models.proposed import NetInfoF


def load_model(args, g, model_name, use_U=True, use_R=True, use_F=True, use_P=True, use_S=True):
    if model_name == 'NetInfoF':
        return NetInfoF(args, g, use_U, use_R, use_F, use_P, use_S)
    else:
        raise NotImplementedError