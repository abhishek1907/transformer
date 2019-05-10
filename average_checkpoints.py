#!/usr/bin/env python3

import argparse
import collections
import torch
import os
import re


def tokenize_de(text):
    return text.split()


def tokenize_en(text):
    return text.split()


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model_state_dict']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k].append(p)

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)
    new_state['model_state_dict'] = averaged_params
    return new_state

if __name__ == '__main__':
    inp = ['fin_nlayer_3model_68_1.5669712440601933.pt', 'fin_nlayer_3model_56_1.568950325066954.pt', 'fin_nlayer_3model_57_1.5699938218032152.pt', 'fin_nlayer_3model_36_1.570087398690216.pt', 'fin_nlayer_3model_47_1.5703490083947433.pt', 'fin_nlayer_3model_73_1.570968065935245.pt', 'fin_nlayer_3model_74_1.57234080141671.pt', 'fin_nlayer_3model_64_1.5727777525580726.pt', 'fin_nlayer_3model_60_1.572860788891704.pt', 'fin_nlayer_3model_63_1.5732345538917587.pt'] 
    # inp = ['nlayer_6model_16k_52_1.6302211958932422.pt', 'nlayer_6model_16k_58_1.6319292444431073.pt','nlayer_6model_16k_56_1.6373203432239316.pt', 'nlayer_6model_16k_39_1.638319762245781.pt', 'nlayer_6model_16k_67_1.6387068006684293.pt', 'nlayer_6model_16k_59_1.6393015875106205.pt', 'nlayer_6model_16k_68_1.6408834286277785.pt', 'nlayer_6model_16k_65_1.642366956791669.pt', 'nlayer_6model_16k_35_1.6424932167017197.pt', 'nlayer_6model_16k_47_1.6426963611173218.pt']
    state = average_checkpoints(inp)
    torch.save(state, 'model.pt')

