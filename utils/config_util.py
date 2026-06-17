"""Small config-access helpers used across eval orchestrators."""


def cfg_get(config, key, default=None):
    """Return config[key] with fallbacks.

    Falls back to `default` if `config` itself is None, the attribute is
    missing, or the stored value is the empty string. The empty-string case
    lets yml configs use `''` as a placeholder that falls through to the
    Python default.
    """
    if config is None:
        return default
    value = getattr(config, key, default)
    if value == "":
        return default
    return value
