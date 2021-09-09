#!/usr/bin/env python

"""


Created: 
"""

import logging, sys, traceback
from typing import Union, Sequence, Optional, Dict, Callable

import numpy as np

from fire.geometry.s_coordinate import get_s_coord_global_r, get_s_coord_path_ds
from fire.misc.utils import filter_kwargs

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

coord = Union[float, np.ndarray]

def write_to_output_format(file_format_plugins: Dict[str, Callable], path_data, image_data, path_names,
                           variable_names_path, variable_names_time, variable_names_image,
                           device_info, meta_data, header_info=None,
                           fn_output=None, path_output=None, raise_on_fail=True, **kwargs_top):
    # TODO: Write all analysis settings eg theo_kwargs
    outputs = {}
    write_plugin_attr = 'write_output'
    for plugin, attrs in file_format_plugins.items():
        if write_plugin_attr in attrs:
            try:
                func = attrs[write_plugin_attr]
                # kwargs order of precedence: kwargs!=None -> Plugin module defaults -> Plugin function defaults
                attrs.update(kwargs_top)
                kwargs = filter_kwargs(attrs, func, remove_from_input=False)
                kws = {k: v for k, v in dict(fn_output=fn_output, path_output=path_output).items() if v is not None}
                kwargs.update(kws)
                outputs[plugin] = func(path_data, image_data, path_names,
                                       variable_names_path, variable_names_time, variable_names_image,
                                       header_info, device_info, meta_data, **kwargs)
            # except KeyError as e:
            #
            except Exception as e:
                if raise_on_fail:
                    raise e
                else:
                    traceback.print_exc()
                    logger.warning(f'Failed to write output data with plugin "{plugin}": {e}')
    return outputs

if __name__ == '__main__':
    pass