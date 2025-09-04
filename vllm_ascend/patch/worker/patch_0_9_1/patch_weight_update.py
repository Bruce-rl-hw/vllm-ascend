#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""
NPU weight update patch - ensures NPU synchronization in weight update utilities.
This patch replaces CUDA synchronization with NPU synchronization in vLLM's weight update functions.
"""

import torch
import functools
from vllm_ascend.utils import logger


def patch_npu_weight_update():
    """Apply NPU synchronization patch for weight update functionality."""
    
    try:
        from vllm.worker import _weight_update
        
        # Store original function
        original_stream_apply = _weight_update.stream_apply_sharded_state
        
        @functools.wraps(original_stream_apply)
        def npu_stream_apply_sharded_state(model, path, pattern=None):
            """NPU-compatible version with proper synchronization."""
            result = original_stream_apply(model, path, pattern)
            # Replace any CUDA sync calls with NPU sync
            torch.npu.synchronize()
            return result
        
        # Apply patch
        _weight_update.stream_apply_sharded_state = npu_stream_apply_sharded_state
        
        logger.debug("NPU weight update synchronization patch applied")
        
    except ImportError:
        logger.debug("vllm.worker._weight_update not available, skipping sync patch")


# Apply the patch
patch_npu_weight_update()
