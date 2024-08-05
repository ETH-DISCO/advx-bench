# https://github.com/wang-research-lab/roz/blob/6b4b7ff9d98a0a6fb4aeb4512859c1a0b16a0138/scripts/natural_distribution_shift/src/registry.py
# https://github.com/wang-research-lab/roz/blob/6b4b7ff9d98a0a6fb4aeb4512859c1a0b16a0138/scripts/natural_distribution_shift/src/inference.py#L115


# figure this out later


# from importlib import import_module
# from pathlib import Path

# MODEL_REGISTRY = dict()


# def register_model(cls):
#     # use @register_model decorator to store
#     print(f"registering {cls.__name__}")
#     MODEL_REGISTRY[cls.__name__] = cls
#     return cls


# def load_models():
#     # not sure whether this actually works
#     models_dir = Path.cwd() / "src" / "models"

#     print(f"loading models from {models_dir}")
#     for f in models_dir.iterdir():
#         if f.suffix == ".py":
#             if "__" not in f.stem:
#                 print(f"loading {f.stem}")
#                 import_module(f"models.{f.stem}")

#     models_subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
#     for subdir in models_subdirs:
#         for f in subdir.iterdir():
#             if f.suffix == ".py":
#                 if "__" not in f.stem:
#                     print(f"loading {subdir.stem}.{f.stem}")
#                     import_module(f"models.{subdir.stem}.{f.stem}")


# def get_model(model_name, **kwargs):
#     if model_name in MODEL_REGISTRY:
#         return MODEL_REGISTRY[model_name](**kwargs)
#     raise ValueError(f"model {model_name} not found in registry")
