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

def _patch_api_server():
    """Add model.safetensors fallback to API server preflight check."""
    try:
        import vllm.entrypoints.openai.api_server as api_module
        original_func = api_module.update_weights_from_disk
        
        async def patched_update_weights_from_disk(raw_request):
            """Auto-detect model.safetensors when no sharded files found."""
            import os
            import glob
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info("=== PATCH: update_weights_from_disk called ===")
            
            try:
                body = await raw_request.json()
                logger.info(f"PATCH: Request body: {body}")
            except Exception as e:
                logger.error(f"PATCH: Failed to parse JSON: {e}")
                return await original_func(raw_request)
            
            path = body.get("path")
            pattern = body.get("pattern")
            logger.info(f"PATCH: path={path}, pattern={pattern}")
            
            # Only patch when no pattern specified and during preflight check failure
            if path and not pattern:
                logger.info("PATCH: No pattern specified, checking for fallback")
                
                try:
                    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
                    default_pattern = ShardedStateLoader.DEFAULT_PATTERN
                    logger.info(f"PATCH: Default pattern: {default_pattern}")
                    
                    wildcard = default_pattern.format(rank="*", part="*")
                    shard_pattern = os.path.join(path, wildcard)
                    logger.info(f"PATCH: Searching for sharded files: {shard_pattern}")
                    
                    shard_files = glob.glob(shard_pattern)
                    logger.info(f"PATCH: Found {len(shard_files)} sharded files: {shard_files}")
                    
                    # Check if no sharded files exist
                    if not shard_files:
                        logger.info("PATCH: No sharded files found, checking model.safetensors")
                        model_safetensors_path = os.path.join(path, "model.safetensors")
                        logger.info(f"PATCH: Checking for: {model_safetensors_path}")
                        
                        if os.path.exists(model_safetensors_path):
                            logger.info("PATCH: model.safetensors found! Setting pattern")
                            # Set pattern and call original
                            body["pattern"] = "model.safetensors"
                            logger.info(f"PATCH: Updated body: {body}")
                            
                            from unittest.mock import MagicMock
                            new_request = MagicMock()
                            new_request.json = lambda: body
                            new_request.app = raw_request.app
                            
                            logger.info("PATCH: Calling original function with model.safetensors pattern")
                            return await original_func(new_request)
                        else:
                            logger.warning(f"PATCH: model.safetensors not found at {model_safetensors_path}")
                    else:
                        logger.info("PATCH: Sharded files found, using original logic")
                        
                except Exception as e:
                    logger.error(f"PATCH: Error in fallback logic: {e}", exc_info=True)
            else:
                logger.info("PATCH: Pattern specified or no path, using original logic")
            
            # Call original function
            logger.info("PATCH: Calling original function")
            return await original_func(raw_request)
        
        api_module.update_weights_from_disk = patched_update_weights_from_disk
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("API server model.safetensors fallback patch applied")
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to patch API server: {e}", exc_info=True)

_patch_api_server()