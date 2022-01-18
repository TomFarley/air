# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created:
"""
import glob
import logging
import os
from typing import Union, Tuple, List, Optional, Dict
from pathlib import Path
from collections import OrderedDict, defaultdict


from fire.interfaces.interfaces import PathList, get_module_from_path_fn
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

plugin_defaults_default = dict(machine='mast_u')

def search_for_plugins(plugin_paths, attributes_requried, attributes_optional=None):
    plugins_all = {}
    info_all = {}

    plugin_paths = make_iterable(plugin_paths)
    for path in plugin_paths:
        # TODO: Catch module not found errors for failed plugins
        plugins, info = get_plugins(path, attributes_requried, attributes_optional=attributes_optional)
        plugins_all.update(plugins)
        info_all.update(info)
    return plugins_all, info_all

def get_plugins(path: Union[Path, str], attributes_requried: Dict[str, str],
                attributes_optional: Optional[Dict[str, str]]=None) -> Tuple[Dict, Dict]:
    """Search directory for python modules containing the requested plugin attributes.

    Note the plugin module should follow the following conventions:
        - Contain at least one required attribute.
        - The first required attribute have a string vlaue that can be used as a name to key the plugin

    Args:
        path                : Directory to search for plugin functions/attributes
        attributes_requried : Dict of required module attributes (i.e. functions, variables) to be extracted of form:
                              {attribute_key_1: attribute_name_1, attribute_key_2: attribute_name_2, ...}
                              The keys are used as a shorthand for looking up the attributes in the returned dict.
        attributes_optional : Dict of optional module attributes (i.e. functions, variables) to be extracted.
                              (see attributes_requried for more details)

    Returns: Two dictionaries (plugins, modules):
             - plugins contains the located plugin attributes, keyed by the value of the first required attribute in
                each
             - modules contains the module object from which each set of plugins was extracted
             e.g.:
             {<attribute_value_1>: {attribute_key_1: attribute_value_1, attribute_key_2: attribute_value_2...},
                           (<my_other_plugin_module>, [plugin_attribute_1, plugin_attribute_2, ...]),
              ...]

    """
    path = Path(path).expanduser().resolve()
    plugins, pulgins_info = defaultdict(dict), defaultdict(dict)
    attributes_requried = make_iterable(attributes_requried)
    attributes_optional = make_iterable(attributes_optional) if attributes_optional is not None else []

    if not path.is_dir():
        logger.warning(f'Plugin search directory does not exist: {str(path)}')
        return plugins, pulgins_info

    # Get possible modules in directory
    file_list = glob.glob(os.path.join(str(path), '*'))
    possible_plugin_modules = []
    for f in file_list:
        if os.path.isdir(f) and os.path.isfile(os.path.join(f, '__init__.py')):
            possible_plugin_modules.append(os.path.join(f, '__init__.py'))
        elif f.endswith('.py'):
            possible_plugin_modules.append(f)

    for path_fn in possible_plugin_modules:
        module = get_module_from_path_fn(path_fn)
        if module is None:
            continue
        plugin_attributes = {}
        info = {'module': module, 'requried': attributes_requried, 'optional': []}
        for attribute_key, attribute_name in attributes_requried.items():
            try:
                attribute_value = getattr(module, attribute_name)
            except AttributeError as e:
                if len(plugin_attributes) == 0:
                    pass
                else:
                    logger.warning(f'Cannot load incomplete plugin "{plugin_attributes}" '
                                   f'missing attribute "{attribute_name}"')
                break
            else:
                plugin_attributes[attribute_key] = attribute_value
        else:
            # Found all required attributes
            for attribute_key, attribute_name in attributes_optional.items():
                try:
                    attribute_value = getattr(module, attribute_name)
                except AttributeError as e:
                    logger.debug(f'Optional plugin "{attribute_name}"" not found in module {module}')
                else:
                    plugin_attributes[attribute_key] = attribute_value
                    info['optional'].append(attribute_name)
            # Follow convention that required attributes always include 'plugin_name'
            plugin_name = plugin_attributes['plugin_name']
            if not isinstance(plugin_name, str):
                raise ValueError(f'Plugin required_attributes do not follow convention of first attribute value being'
                                 f'string name of plugin: plugin_name_attribute={plugin_name}, module={path_fn}')
            # Update default dict to enable combining of plugin functionality from multiple sources
            plugins[plugin_name].update(plugin_attributes)
            pulgins_info[plugin_name].update(info)
    # plugin = ([module_name, path_fn, ''.join(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))])

    return plugins, pulgins_info

def get_compatible_plugins(plugin_paths: PathList,
                           attributes_required: List[str], attributes_optional: Optional[List[str]]=None,
                           plugin_filter: Optional[List[str]]=None, plugins_required: Optional[List[str]]=None,
                           plugin_type: str= '', base_paths: Union[dict, tuple]=()) -> Tuple[Dict, Dict]:
    """Return a nested dict of plugin functions and attributes found in the supplied paths with the required attributes.

    Args:
        plugin_paths        : Paths in which to locate plugin modules for reading movie data
        attributes_required : Attributes that must be present in a plugin module for it to be loaded
        attributes_optional : Optional attributes that will be loaded in addition to the required plugin attributes
        plugin_filter       : Names of subset of plugins to return that are compatible with your application
        plugins_required    : Names of plugins that are required. An exception is raised if not all found.
        plugin_type         : (Optional) Name of plugin type used in logging messages
        base_paths          : Path roots to substitute into path format strings

    Returns: Nested dict of movie plugins with key structure:
             plugins[<plugin_name>]['meta']         - function returning meta data for movie
             plugins[<plugin_name>]['data']         - function returning frame data for movie
             plugins[<plugin_name>]['plugin_info']  - dict of information about the plugin

    """
    if attributes_optional is None:
        attributes_optional = {}

    base_paths = dict(base_paths)
    plugin_paths = [str(p).format(**base_paths) for p in plugin_paths]
    # plugins_all = get_movie_plugins(plugin_paths)

    plugins, info = search_for_plugins(plugin_paths, attributes_required,
                                              attributes_optional=attributes_optional)
    logger.info(f'Located {plugin_type} plugins for: {", ".join(list(plugins.keys()))} (filter: {plugin_filter})')
    if plugins_required is not None:
        missing = []
        for plugin in make_iterable(plugins_required):
            if plugin not in plugins:
                missing.append(plugin)
        if missing:
            raise FileNotFoundError(f'Failed to locate required {plugin_type} plugins {missing} in paths'
                                    f'(NOTE: could be due to import/syntax error in file):\n'
                                    f'{plugin_paths}')
    if plugin_filter is not None:
        # Return filtered plugins in order specified (in json config file)
        plugin_filter = make_iterable(plugin_filter)
        plugins = OrderedDict([(key, plugins[key]) for key in plugin_filter if key in plugins.keys()])
        info = OrderedDict([(key, info[key]) for key in plugin_filter if key in info.keys()])
    return plugins, info

def call_plugin_func(plugins_dict, plugin_func_name, args=(), kwargs=(), dummy_ouput=None,
                     verbose_dummy_func=True, raise_on_fail=True):
    args = make_iterable(args)
    kwargs = dict(kwargs)

    dummy_func = dummy_function(plugin_func_name, dummy_ouput, verbose=verbose_dummy_func)

    func = plugins_dict.get(plugin_func_name, dummy_func)

    if func is dummy_func:
        message = f'Could not find plugin "{plugin_func_name}" in plugin dict: {list(plugins_dict.keys())}'
        if raise_on_fail:
            raise ValueError(message)
        else:
            logger.warning(message)

    out = func(*args, **kwargs)
    return out

def get_and_call_plugin_func(plugin_func_name, plugins_dict=None, plugin_type='machine', plugin_defaults=None,
                             kwargs_plugin=(), raise_on_fail=True):
    from fire.plugins.plugins_machine import get_machine_plugins

    kwargs_plugin = dict(kwargs_plugin)

    if plugins_dict is None:
        if plugin_type == 'machine':
            plugin_defaults = plugin_defaults_default if plugin_defaults is None else plugin_defaults
            machine = plugin_defaults.get('machine', 'mast_u')
            plugins_dict = get_machine_plugins(machine=machine)

    output = call_plugin_func(plugins_dict=plugins_dict, plugin_func_name=plugin_func_name, kwargs=kwargs_plugin,
                              raise_on_fail=raise_on_fail)

    return output

def dummy_function(name, dummy_ouput, verbose=True):
    def _dummy_func(*args, **kwargs):
        if verbose:
            logger.warning(f'Failed to call plugin function "{name}" with {args}, {kwargs}')
        return dummy_ouput
    return _dummy_func

if __name__ == '__main__':
    pass