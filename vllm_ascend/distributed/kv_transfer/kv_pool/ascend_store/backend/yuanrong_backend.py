import hashlib
import os
import re

import torch
from vllm.config import ParallelConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend


def split_host_port(addr, default_port=None):
    """
    Splits an address string into (host, port).
    Supports formats: '127.0.0.1:80', 'localhost:8080', '192.168.1.1'
    Note: This version is designed for IPv4 and hostnames only.
    """
    if ":" in addr:
        # Split from the right side exactly once to separate host and port
        host, port_str = addr.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            # If the part after the colon is not a valid integer, 
            # treat the entire string as the host.
            host = addr
            port = default_port
    else:
        # No colon found, return host with the default port
        host = addr
        port = default_port
        
    return host, port


class YuanrongBackend(Backend):

    _DS_KEY_MAX_LEN = 255
    _DS_KEY_ALLOWED_PATTERN = re.compile(
        r"^[a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]+$")
    _DS_KEY_INVALID_CHAR_PATTERN = re.compile(
        r"[^a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]")
    _DS_KEY_HASH_SUFFIX_LEN = 16

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import (  # type: ignore
                HeteroClient, Blob, DeviceBlobList)
            from yr.datasystem.kv_client import SetParam  # type: ignore
            from yr.datasystem.object_client import WriteMode  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Please install openyuanrong-datasystem to use the "
                "yuanrong datasystem backend.") from exc

        worker_addr = os.getenv("DS_WORKER_ADDR", "")
        host, port = split_host_port(worker_addr)
        enable_exclusive_connection = bool(
            int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0")))
        enable_remote_h2d = bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0")))
        self._hetero_client = HeteroClient(
            host,
            int(port),
            enable_exclusive_connection=enable_exclusive_connection,
            enable_remote_h2d=enable_remote_h2d,
        )
        self._hetero_client.init()
        self.rank = parallel_config.rank
        self._ds_device_id = None
        self._ds_blob_cls = Blob
        self._ds_blob_list_cls = DeviceBlobList
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

    def _normalize_ds_keys(self, keys: list[str]) -> list[str]:
        normalized: list[str] = []
        for key in keys:
            if (len(key) <= self._DS_KEY_MAX_LEN
                    and self._DS_KEY_ALLOWED_PATTERN.match(key)):
                normalized.append(key)
                continue

            sanitized = self._DS_KEY_INVALID_CHAR_PATTERN.sub("_", key)
            hash_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            suffix = f"__{hash_digest[:self._DS_KEY_HASH_SUFFIX_LEN]}"
            max_prefix_len = self._DS_KEY_MAX_LEN - len(suffix)
            normalized.append(sanitized[:max_prefix_len] + suffix)
        return normalized

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)
        self._ds_device_id = int(torch.npu.current_device())
    
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    def _ds_make_blob_list(self, addrs: list[int],
                           sizes: list[int]) -> "DeviceBlobList":
        if len(addrs) != len(sizes):
            raise ValueError("Address list and size list length mismatch.")
        if self._ds_device_id is None:
            logger.error(
                "Device id is not set. Call set_device() before using the "
                "yuanrong backend.")
            raise RuntimeError("Yuanrong backend device id is not initialized.")
        blobs = [
            self._ds_blob_cls(addr, size)  # type: ignore[misc]
            for addr, size in zip(addrs, sizes)
        ]
        return self._ds_blob_list_cls(  # type: ignore[misc]
            self._ds_device_id, blobs)

    def exists(self, keys: list[str]) -> list[int]:
        try:
            keys = self._normalize_ds_keys(keys)
            exists = self._hetero_client.exist(keys)  # type: ignore[union-attr]
            return [1 if value else 0 for value in exists]
        except Exception as exc:
            logger.error("Failed to check keys %s: %s", keys, exc)
            return [0] * len(keys)

    def get(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        try:
            keys = self._normalize_ds_keys(keys)
            blob_lists = [
                self._ds_make_blob_list(addr_list, size_list)
                for addr_list, size_list in zip(addrs, sizes)
            ]
            failed_keys = self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                keys, blob_lists, 0)
            for key in failed_keys:
                logger.error("Failed to get key %s", key)
        except Exception as exc:
            logger.error("Failed to get keys %s: %s", keys, exc)

    def put(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        try:
            keys = self._normalize_ds_keys(keys)
            blob_lists = [
                self._ds_make_blob_list(addr_list, size_list)
                for addr_list, size_list in zip(addrs, sizes)
            ]
            self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                keys, blob_lists, self._ds_set_param)
        except Exception as exc:
            logger.error("Failed to put keys %s: %s", keys, exc)