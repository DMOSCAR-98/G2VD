"""
Config-driven instantiation: resolve "module.Class" strings and build objects from dicts.

Used across the project for YAML/OmegaConf targets (detector, CFIPipeline, augmentations).
Follows Google-style docstrings; see .cursor/rules/python-docstrings.mdc for project norms.
"""
import importlib
from typing import Any, Dict


def get_obj_from_str(
    string: str,
    reload: bool = False,
) -> Any:
    """
    Get a class or function object from a string path.

    The string should be in the format "module.path.ClassName" or
    "module.path.function_name". This function imports the module and
    returns the specified class or function.

    Args:
        string (str): Full path to the class or function in format
            "module.path.ClassName" or "module.path.function_name".
        reload (bool, optional): Whether to reload the module before
            getting the object. Useful for development when modules may
            have changed. Defaults to False.

    Returns:
        Any: The class or function object specified by the string path.

    Examples:
        >>> # Get a PyTorch class
        >>> Linear = get_obj_from_str("torch.nn.Linear")

        >>> # Get a custom class with reload
        >>> MyClass = get_obj_from_str("my_module.MyClass", reload=True)
    """
    module, cls = string.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Dict[str, Any]) -> Any:
    """
    Instantiate an object from a configuration dictionary.

    The configuration dictionary should contain a "target" key specifying the
    class to instantiate, and optionally a "params" key containing parameters
    to pass to the constructor.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - target (str): Full class path in format "module.ClassName".
            - params (Dict[str, Any], optional): Dictionary of parameters to pass
                to the constructor. Defaults to empty dictionary.

    Returns:
        Any: Instantiated object of the specified class.

    Raises:
        KeyError: If "target" key is missing from config.

    Examples:
        >>> config = {
        ...     "target": "torch.nn.Linear",
        ...     "params": {"in_features": 512, "out_features": 256},
        ... }
        >>> layer = instantiate_from_config(config)
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))
