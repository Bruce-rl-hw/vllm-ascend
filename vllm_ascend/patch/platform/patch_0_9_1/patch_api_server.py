#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""
Patch API Server to auto-detect model.safetensors when no sharded files found.
"""

import functools
from vllm_ascend.utils import logger


def patch_api_server_weight_update():
    """Add model.safetensors fallback to API server preflight check."""
    
    try:
        import vllm.entrypoints.openai.api_server as api_module
        
        # Store original function
        original_func = api_module.update_weights_from_disk
        
        @functools.wraps(original_func)
        async def patched_update_weights_from_disk(raw_request):
            """Auto-detect model.safetensors when no sharded files found."""
            import os
            import glob
            
            logger.info("=== PATCH: update_weights_from_disk called ===")
            
            try:
                body = await raw_request.json()
                path = body.get("path")
                pattern = body.get("pattern")
                
                logger.info(f"PATCH: path={path}, pattern={pattern}")
                
                # Only patch when no pattern is specified
                if path and not pattern:
                    # Check for sharded files first
                    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
                    default_pattern = ShardedStateLoader.DEFAULT_PATTERN
                    wildcard = default_pattern.format(rank="*", part="*")
                    shard_pattern = os.path.join(path, wildcard)
                    shard_files = glob.glob(shard_pattern)
                    
                    logger.info(f"PATCH: Found {len(shard_files)} sharded files")
                    
                    # If no sharded files, check for model.safetensors
                    if not shard_files:
                        model_safetensors_path = os.path.join(path, "model.safetensors")
                        if os.path.exists(model_safetensors_path):
                            logger.info("PATCH: model.safetensors found! Setting pattern")
                            # Modify the request body to include the pattern
                            body["pattern"] = "model.safetensors"
                            
                            # Create a new request with the modified body
                            from unittest.mock import AsyncMock, MagicMock
                            new_request = MagicMock()
                            new_request.json = AsyncMock(return_value=body)
                            new_request.app = raw_request.app
                            
                            return await original_func(new_request)
                        else:
                            logger.warning(f"PATCH: model.safetensors not found at {model_safetensors_path}")
                
            except Exception as e:
                logger.error(f"PATCH: Error in preprocessing: {e}", exc_info=True)
            
            # Call original function for all other cases
            return await original_func(raw_request)
        
        # Apply patch
        api_module.update_weights_from_disk = patched_update_weights_from_disk

        # 强制替换FastAPI路由的endpoint，确保补丁生效
        router = getattr(api_module, "router", None)
        if router is not None and hasattr(router, "routes"):
            for route in router.routes:
                if getattr(route, "path", None) == "/update-weights-from-disk":
                    route.endpoint = patched_update_weights_from_disk
                    if hasattr(route, "dependant") and hasattr(route.dependant, "call"):
                        route.dependant.call = patched_update_weights_from_disk
                    logger.info("PATCH: FastAPI route endpoint replaced")

        logger.info("API server model.safetensors fallback patch applied")

    except Exception as e:
        logger.error(f"Failed to patch API server: {e}", exc_info=True)


# Apply the patch
patch_api_server_weight_update()