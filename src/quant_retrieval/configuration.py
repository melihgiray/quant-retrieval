"""Read experiment settings without silently replacing duplicate YAML fields."""

from pathlib import Path

import yaml


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                None, None, "configuration keys must be strings", key_node.start_mark
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate configuration field: {key}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(), Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value
