# Standard
import hashlib
import os
import re
import time
from types import SimpleNamespace

# Third Party
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.utils import get_ip, logger, split_host_port

from vllm_ascend.distributed.mooncake.config_data import MooncakeEngineKey
from vllm_ascend.distributed.mooncake.transfer_engine import get_global_te

from .config_data import MooncakeStoreConfig

METADATA_BYTES_LEN = 24
BASE_PORT = int(os.getenv("VLLM_BASE_PORT", "8790"))
_DEBUG_LOG = bool(os.getenv("DS_DEBUG_LOG", False))
_ENBALE_RH2D = bool(os.getenv("DS_ENABLE_RH2D", False))
_DS_KEY_ALLOWED_RE = re.compile(
    r"^[a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]*$")
_DS_KEY_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]")
_DS_KEY_MAX_LEN = 255
_DS_KEY_HASH_LEN = 16


class Mooncakestore():

    def __init__(self, parallel_config: ParallelConfig):
        self._use_datasystem = self._is_datasystem_backend()
        self._profile_store = os.getenv("MOONCAKE_STORE_PROFILE", "0") == "1"
        if self._use_datasystem:
            self._init_datasystem_store(parallel_config)
            return
        try:
            from mooncake.store import (  # type: ignore
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        self._replicate_config_cls = ReplicateConfig
        device_id = self._get_device_id(parallel_config)
        self.config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()
        if self.config.protocol == "ascend" and not self.config.use_ascend_direct:
            local_hostname = get_ip() + ":" + str(BASE_PORT + int(device_id)) + \
                ":npu_" + str(device_id)
            ret = self.store.setup(local_hostname, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address)
        else:
            local_hostname = get_ip()
            transfer_engine = get_global_te(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(
                transfer_engine.get_rpc_port())
            ret = self.store.setup(self.local_seg, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address,
                                   transfer_engine.get_engine())
        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

    @staticmethod
    def _is_datasystem_backend() -> bool:
        backend = os.getenv("MOONCAKE_STORE_BACKEND", "").strip().lower()
        return backend in {"datasystem", "yuanrong", "ds", "yr"}

    def _get_device_id(self, parallel_config: ParallelConfig) -> int:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = parallel_config.tensor_parallel_size
        dp_rank = parallel_config.data_parallel_rank_local
        all_device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
        if not all_device_ids:
            device_ids_list = list(
                range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        else:
            device_ids_list = list(map(int, all_device_ids.split(',')))
        assert len(device_ids_list) > tp_rank
        return device_ids_list[tp_rank]

    def _init_datasystem_store(self, parallel_config: ParallelConfig) -> None:
        try:
            from yr.datasystem.hetero_client import (  # type: ignore
                HeteroClient, Blob, DeviceBlobList)
            from yr.datasystem.kv_client import SetParam  # type: ignore
            from yr.datasystem.object_client import WriteMode  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install openyuanrong-datasystem to use the "
                "datasystem backend.") from e

        worker_addr = os.getenv("DS_WORKER_ADDR", "")
        if not worker_addr:
            raise ValueError(
                "DS_WORKER_ADDR is not set for datasystem backend.")
        host, port = split_host_port(worker_addr)
        logger.info(f"Enable RH2D: {_ENBALE_RH2D}")
        self._ds_client = HeteroClient(host, int(port), enable_remote_h2d=_ENBALE_RH2D)
        self._ds_client.init()
        self._ds_blob_cls = Blob
        self._ds_blob_list_cls = DeviceBlobList
        import torch
        self._ds_device_id = int(torch.npu.current_device())
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT
        self.config = SimpleNamespace(use_ascend_direct=True)

    @staticmethod
    def _normalize_ds_key(key: str) -> str:
        if (_DS_KEY_ALLOWED_RE.match(key)
                and len(key) <= _DS_KEY_MAX_LEN):
            return key
        safe = _DS_KEY_SANITIZE_RE.sub("_", key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        suffix = "__" + digest[:_DS_KEY_HASH_LEN]
        max_prefix_len = _DS_KEY_MAX_LEN - len(suffix)
        if max_prefix_len < 0:
            max_prefix_len = 0
        return safe[:max_prefix_len] + suffix

    @classmethod
    def _normalize_ds_keys(cls, keys: list[str]) -> list[str]:
        return [cls._normalize_ds_key(key) for key in keys]

    def _ds_make_blob_list(self, addrs: list[int],
                           sizes: list[int]) -> "DeviceBlobList":
        if len(addrs) != len(sizes):
            raise ValueError("Address list and size list length mismatch.")
        blobs = [
            self._ds_blob_cls(addr, size)
            for addr, size in zip(addrs, sizes)
        ]
        return self._ds_blob_list_cls(self._ds_device_id, blobs)

    def _log_store_latency(self, op_name: str, start_time: float,
                           keys: list[str], addrs: list[list[int]],
                           sizes: list[list[int]]) -> None:
        if not self._profile_store:
            return
        total_buffers = sum(len(addr_list) for addr_list in addrs)
        total_bytes = sum(sum(size_list) for size_list in sizes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        backend = "datasystem" if self._use_datasystem else "mooncake"
        logger.info(
            "Store backend=%s op=%s completed in %.3f ms "
            "(keys=%d, buffers=%d, bytes=%d)",
            backend,
            op_name,
            elapsed_ms,
            len(keys),
            total_buffers,
            total_bytes,
        )

    def exists(self, key: MooncakeEngineKey) -> bool:
        if self._use_datasystem:
            try:
                key_str = key.to_string()
                res = self._ds_client.exist(
                    self._normalize_ds_keys([key_str]))
                if _DEBUG_LOG:
                    logger.info(f"exists: {key_str}:{res[0]}")
                return bool(res[0]) if res else False
            except Exception as e:
                logger.error(f"Failed to check key {key.to_string()}: {e}")
                return False
        return self.store.is_exist(key.to_string()) == 1

    def batch_exists(self, keys: list[str]) -> list[int]:
        if self._use_datasystem:
            if not keys:
                return []
            try:
                ds_keys = self._normalize_ds_keys(keys)
                exists = self._ds_client.exist(ds_keys)
                if _DEBUG_LOG:
                    output = "batch_exists: " + ", ".join([f"{a}:{b}" for a, b in zip(ds_keys, exists)])
                    logger.info(output)
                return [1 if value else 0 for value in exists]
            except Exception as e:
                logger.error(f"Failed to check keys {keys}: {e}")
                return [0] * len(keys)
        return self.store.batch_is_exist(keys)

    def register_buffer(self, ptr, length):
        if self._use_datasystem:
            return None
        return self.store.register_buffer(ptr, length)

    def get_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        start_time = time.perf_counter() if self._profile_store else 0.0
        if self._use_datasystem:
            if not keys:
                return None
            try:
                ds_keys = self._normalize_ds_keys(keys)
                blob_lists = [
                    self._ds_make_blob_list(addr_list, size_list)
                    for addr_list, size_list in zip(addrs, sizes)
                ]
                failed_keys = self._ds_client.mget_h2d(
                    ds_keys, blob_lists, 0)
                failed_set = set(failed_keys)
                res = [0 if key not in failed_set else -1 for key in ds_keys]
                if _DEBUG_LOG:
                    output = "get_batch: " + ", ".join([f"{a}:{b}" for a, b in zip(ds_keys, res)])
                    logger.info(output)
                return res
            except Exception as e:
                logger.error(f"Failed to get key {keys}. {e}")
                return [-1] * len(keys)
            finally:
                self._log_store_latency("get_batch", start_time, keys, addrs,
                                        sizes)
        try:
            res = self.store.batch_get_into_multi_buffers(
                keys, addrs, sizes, True)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {keys}. {e}")
        finally:
            self._log_store_latency("get_batch", start_time, keys, addrs, sizes)

    def put_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        start_time = time.perf_counter() if self._profile_store else 0.0
        if self._use_datasystem:
            if not keys:
                return None
            try:
                ds_keys = self._normalize_ds_keys(keys)
                blob_lists = [
                    self._ds_make_blob_list(addr_list, size_list)
                    for addr_list, size_list in zip(addrs, sizes)
                ]
                self._ds_client.mset_d2h(ds_keys, blob_lists,
                                         self._ds_set_param)
                if _DEBUG_LOG:
                    output = "put_batch: " + ", ".join([f"{a}" for a in ds_keys])
                    logger.info(output)
            except Exception as e:
                logger.error(f"Failed to put key {keys},error:{e}")
            finally:
                self._log_store_latency("put_batch", start_time, keys, addrs,
                                        sizes)
            return None
        try:
            config = self._replicate_config_cls()
            config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = True
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to put key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {keys},error:{e}")
        finally:
            self._log_store_latency("put_batch", start_time, keys, addrs, sizes)

    def get(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        """Get a single key-value pair from store.
        
        Note: This method is used when use_ascend_direct=False.
        When use_ascend_direct=True, use get_batch() instead.
        """
        start_time = time.perf_counter() if self._profile_store else 0.0
        expect_res = sum(size)
        key_str = key.to_string()
        if self._use_datasystem:
            try:
                blob_list = self._ds_make_blob_list(addr, size)
                ds_key = self._normalize_ds_key(key_str)
                failed_keys = self._ds_client.mget_h2d([ds_key],
                                                      [blob_list], 0)
                if failed_keys:
                    logger.error(f"Failed to get key: [{key_str}] .")
                    return [-1]
                if _DEBUG_LOG:
                    logger.info("get: {ds_key}")
                return [expect_res]
            except Exception:
                logger.error(f"Failed to get key: [{key_str}] .")
                return [-1]
            finally:
                self._log_store_latency("get", start_time, [key_str], [addr],
                                        [size])

        res = None
        try:
            # Try ascend direct method if available and configured
            if self.config.use_ascend_direct and hasattr(self.store, 'batch_get_into_ascend'):
                res = self.store.batch_get_into_ascend(key_str, addr, size)
            else:
                # Fallback to multi_buffers method
                res = self.store.batch_get_into_multi_buffers(
                    [key_str], [addr], [size], True)
                res = res[0] if res else None
            
            if res is None:
                logger.error(f"Failed to get key: [{key_str}], method returned None")
                return [-1]
            elif isinstance(res, (list, tuple)) and len(res) > 0:
                if res[0] != expect_res:
                    logger.error(
                        f"Failed to get key: [{key_str}], expected {expect_res}, got {res[0]}")
        except AttributeError as e:
            logger.error(
                f"Failed to get key: [{key_str}], method not available: {e}. "
                f"use_ascend_direct={self.config.use_ascend_direct}")
            return [-1]
        except Exception as e:
            logger.error(
                f"Failed to get key: [{key_str}], error: {e}, error_type={type(e).__name__}")
            return [-1]
        finally:
            self._log_store_latency("get", start_time, [key_str], [addr],
                                    [size])

        return res if res is not None else [-1]

    def put(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        """Put a single key-value pair to store.
        
        Note: This method is used when use_ascend_direct=False.
        When use_ascend_direct=True, use put_batch() instead.
        """
        start_time = time.perf_counter() if self._profile_store else 0.0
        key_str = key.to_string()
        if self._use_datasystem:
            try:
                blob_list = self._ds_make_blob_list(addr, size)
                ds_key = self._normalize_ds_key(key_str)
                self._ds_client.mset_d2h([ds_key], [blob_list],
                                         self._ds_set_param)
                if _DEBUG_LOG:
                    logger.info("put: {ds_key}")
                return [0]
            except Exception:
                logger.error(f"Failed to put key {key_str}.")
                return [-1]
            finally:
                self._log_store_latency("put", start_time, [key_str], [addr],
                                        [size])

        ret = None
        try:
            # Try ascend direct method if available and configured
            if self.config.use_ascend_direct and hasattr(self.store, 'batch_put_from_ascend'):
                ret = self.store.batch_put_from_ascend(key_str, addr, size)
            else:
                # Fallback to multi_buffers method
                config = self._replicate_config_cls()
                config.preferred_segment = getattr(self, 'local_seg', None)
                config.prefer_alloc_in_same_node = True
                ret = self.store.batch_put_from_multi_buffers(
                    [key_str], [addr], [size], config)
                ret = ret[0] if ret else None
            
            if ret is None:
                logger.error(f"Failed to put key {key_str}, method returned None")
                return [-1]
            elif isinstance(ret, (list, tuple)) and len(ret) > 0:
                if ret[0] != 0:
                    logger.error(
                        f"Failed to put key {key_str}, return code: {ret[0]}")
        except AttributeError as e:
            logger.error(
                f"Failed to put key {key_str}, method not available: {e}. "
                f"use_ascend_direct={self.config.use_ascend_direct}")
            return [-1]
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str}, error: {e}, error_type={type(e).__name__}")
            return [-1]
        finally:
            self._log_store_latency("put", start_time, [key_str], [addr],
                                    [size])

        return ret if ret is not None else [-1]

    def close(self):
        if self._use_datasystem:
            logger.info("Closed the datasystem store connection")
            return
        self.store.close()
        logger.info("Closed the mooncake store connection")
