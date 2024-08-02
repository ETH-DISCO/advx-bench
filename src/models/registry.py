MODEL_REGISTRY = dict()


def store_model(cls):
    # use @register_model_class decorator to store
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


def get_model(model_name, **kwargs):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**kwargs)
    raise ValueError(f"model {model_name} not found in registry")
