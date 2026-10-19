"""Multi-backend GPU memory allocator benchmark (UMD pool / KMD direct).

Backends: cuda / hip / xpu (--backend; "auto" probes cuda/hip/xpu in order).

Measured capabilities (against the KMD kernel-driver allocator):
  1. Baseline alloc: cold/hot latency and write bandwidth per block size.
  2. Anti-fragmentation: carve from one segment -> alternate frees -> merge/reclaim.
  3. Alloc under severe fragmentation: random fill + random free, growing contiguous
     allocs, small-block steady-state bench, cleanup recovery.

Measurement layers:
  Default (UMD pool): torch caching allocator; pool reuse/merge is UMD behavior,
    only cold paths reach KMD.
  --direct: one KMD alloc per call. cuda/hip: ctypes cudaMalloc/hipMalloc; xpu:
    Level Zero zeMemAllocDevice (libze_loader.so.1). The fragmentation tests need
    the direct API to build KMD-level fragmentation; without it they fall back to
    UMD pool semantics.
  --ze-uncached (xpu): adds BIAS_UNCACHED to the alloc flags (GPU UC caching policy
    for cache A/B, not a pool bypass).
  --contig (xpu): KMD allocs via zePhysicalMemCreate + zeVirtualMemMap -- one physical
    object per alloc (contiguous physical backing, no page scatter), the
    fragmentation-sensitive mode: scattered free space can no longer serve large
    requests. cuda/hip need no flag (cudaMalloc/hipMalloc are contiguous-demanding).

Usage / A-B (comparison is manual; the script never auto-compares):
  python mem_perf.py --backend xpu
  python mem_perf.py --backend cuda --direct --json direct.json
  python mem_perf.py --backend xpu --direct --contig --ratio 1.0
  Diff the JSON reports across allocator/driver configs. Random sequences come from
  the --seed PRNG stream; the plan (fill sizes / freed indices) is recorded in JSON
  (severe_frag.plan_fill_sizes / plan_freed_indices) for audit.

Allocator config must be set before python starts:
  cuda: PYTORCH_CUDA_ALLOC_CONF; uncached bypass: PYTORCH_NO_CUDA_MEMORY_CACHING=1
  hip:  PYTORCH_HIP_ALLOC_CONF
  xpu:  torch 2.8.0+xpu XPUCachingAllocator parses no allocator config (verified);
        true direct alloc exists via Level Zero ctypes (--direct).

Memory pressure: --ratio scales the test budget against driver free memory; --ratio 1.0
  fills ALL free memory for the fragmentation tests (no large contiguous region left).
  With ample device memory, UMD pool fragmentation does not OOM (KMD supplies new
  segments); it only grows reserved. A larger --bench-mb (e.g. 32) amplifies pool
  split/search overhead.
"""

import argparse
import ctypes
import gc
import json
import os
import random
import time

import torch

# Legacy IPEX environments need this import to register the XPU backend
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass

MB = 1024**2
GB = 1024**2 * 1024
LADDER_MB = [16, 32, 64, 128, 256, 512, 1024, 2048]     # frag ladder (growing contiguous allocs)
FILL_SIZES_MB = [8, 16, 24, 32, 48, 64, 96]             # frag random fill sizes
SIZES_MB = [4, 32, 128, 512]                            # baseline block sizes
ENV_KEYS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_XPU_ALLOC_CONF",
)
WIDTH = 62  # console rule width for banner / section headers

ZE_ALLOC_FLAGS = 0  # set to ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED (0x2) on --ze-uncached


class RawOOM(Exception):
    """Raw driver allocation rejected (KMD OOM)."""


# ========= Raw driver allocator (--direct; bypasses the UMD pool) =========
class RawDriver:
    """ctypes bindings to the driver user-mode library; each alloc/free lands in KMD, no pooling."""

    LIBS = {
        "cuda": ["libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"],
        "hip": ["libhipamd.so", "libhipamd.so.1", "libhipamd.so.0"],
    }
    FNS = {
        "cuda": ("cudaMalloc", "cudaFree", "cudaDeviceSynchronize"),
        "hip": ("hipMalloc", "hipFree", "hipDeviceSynchronize"),
    }

    def __init__(self, flavor: str):
        lib = None
        for cand in self.LIBS[flavor]:
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise OSError(f"{flavor} driver library not found: {self.LIBS[flavor]}")
        alloc_name, free_name, sync_name = self.FNS[flavor]
        self._alloc_fn = getattr(lib, alloc_name)
        self._alloc_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._alloc_fn.restype = ctypes.c_int
        self._free_fn = getattr(lib, free_name)
        self._free_fn.argtypes = [ctypes.c_void_p]
        self._free_fn.restype = ctypes.c_int
        self._sync_fn = getattr(lib, sync_name)
        self._sync_fn.restype = ctypes.c_int
        self.desc = f"{alloc_name}(ctypes)"
        # self-check: 1MB alloc/free
        self.free(self.alloc(MB))

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        rc = self._alloc_fn(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if rc != 0 or not ptr.value:
            raise RawOOM(f"{self.desc} rc={rc}")
        return ptr.value

    def free(self, ptr: int):
        rc = self._free_fn(ctypes.c_void_p(ptr))
        if rc != 0:
            print(f"  ⚠️ {self.desc} free rc={rc}")

    def sync(self):
        rc = self._sync_fn()
        if rc != 0:
            print(f"  ⚠️ {self.desc} sync rc={rc}")


# ========= Level Zero direct (xpu KMD direct path) =========
class LevelZeroDriver:
    """ctypes bindings to the Level Zero loader: zeMemAllocDevice hits KMD per call, no pooling.

    Enum values from the local /usr/local/include/level_zero/ze_api.h. A pure alloc/free
    benchmark needs no command queue/kernel; the alloc context is self-built here,
    independent of torch's. Allocations land on the first GPU device (KMD_free is
    observed per single device on multi-GPU systems).
    """

    ZE_SUCCESS = 0
    ZE_OOM_DEVICE = 0x70000003
    ZE_OOM_HOST = 0x70000002
    ZE_INIT_FLAG_GPU_ONLY = 1
    ZE_STYPE_CONTEXT_DESC = 0xD
    ZE_STYPE_DEVICE_MEM_ALLOC_DESC = 0x15
    ZE_STYPE_DEVICE_PROPERTIES = 0x3
    ZE_STYPE_PHYSICAL_MEM_DESC = 0x20
    ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE = 0x1
    ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1
    ZE_DEVICE_TYPE_GPU = 1

    class _ContextDesc(ctypes.Structure):
        _fields_ = [("stype", ctypes.c_int32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]

    class _MemAllocDesc(ctypes.Structure):
        _fields_ = [
            ("stype", ctypes.c_int32),
            ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32),
            ("ordinal", ctypes.c_uint32),
        ]

    class _PhysicalMemDesc(ctypes.Structure):
        _fields_ = [
            ("stype", ctypes.c_int32),
            ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32),
            ("size", ctypes.c_size_t),
        ]

    class _DeviceProps(ctypes.Structure):
        # only "type" is read; anonymous buffer pads the tail beyond full struct size
        _fields_ = [
            ("stype", ctypes.c_int32),
            ("pNext", ctypes.c_void_p),
            ("type", ctypes.c_int32),
            ("vendorId", ctypes.c_uint32),
            ("deviceId", ctypes.c_uint32),
            ("_pad", ctypes.c_uint8 * 1024),
        ]

    def __init__(self, contig: bool = False):
        lib = None
        for cand in ("libze_loader.so.1", "libze_loader.so"):
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise OSError("Level Zero loader not found: libze_loader.so.1")
        lib.zeInit.argtypes = [ctypes.c_uint32]
        lib.zeInit.restype = ctypes.c_int
        lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
        lib.zeDriverGet.restype = ctypes.c_int
        lib.zeDeviceGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
        lib.zeDeviceGet.restype = ctypes.c_int
        lib.zeDeviceGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(self._DeviceProps)]
        lib.zeDeviceGetProperties.restype = ctypes.c_int
        lib.zeContextCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(self._ContextDesc), ctypes.POINTER(ctypes.c_void_p)]
        lib.zeContextCreate.restype = ctypes.c_int
        lib.zeMemAllocDevice.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(self._MemAllocDesc), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.zeMemAllocDevice.restype = ctypes.c_int
        lib.zeMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.zeMemFree.restype = ctypes.c_int
        lib.zeDeviceSynchronize.argtypes = [ctypes.c_void_p]
        lib.zeDeviceSynchronize.restype = ctypes.c_int
        if contig:
            for fn in (
                "zeVirtualMemQueryPageSize", "zeVirtualMemReserve", "zeVirtualMemMap",
                "zeVirtualMemUnmap", "zeVirtualMemFree", "zePhysicalMemCreate", "zePhysicalMemDestroy",
            ):
                if not hasattr(lib, fn):
                    raise OSError(f"L0 loader lacks {fn} (loader too old for the physical-memory mode)")
                getattr(lib, fn).restype = ctypes.c_int
            lib.zeVirtualMemQueryPageSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
            lib.zeVirtualMemReserve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            lib.zeVirtualMemMap.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            lib.zeVirtualMemUnmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.zeVirtualMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.zePhysicalMemCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(self._PhysicalMemDesc), ctypes.POINTER(ctypes.c_void_p)]
            lib.zePhysicalMemDestroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.contig = contig
        self._lib = lib

        self._check(lib.zeInit(self.ZE_INIT_FLAG_GPU_ONLY), "zeInit")
        nd = ctypes.c_uint32(0)
        self._check(lib.zeDriverGet(ctypes.byref(nd), None), "zeDriverGet")
        if nd.value == 0:
            raise OSError("Level Zero enumerated no driver (check /dev/dri and GPU driver)")
        drivers = (ctypes.c_void_p * nd.value)()
        self._check(lib.zeDriverGet(ctypes.byref(nd), drivers), "zeDriverGet")

        self.device = None
        for drv in drivers:
            n = ctypes.c_uint32(0)
            self._check(lib.zeDeviceGet(drv, ctypes.byref(n), None), "zeDeviceGet")
            if n.value == 0:
                continue
            devs = (ctypes.c_void_p * n.value)()
            self._check(lib.zeDeviceGet(drv, ctypes.byref(n), devs), "zeDeviceGet")
            for dev in devs:
                props = self._DeviceProps()
                props.stype = self.ZE_STYPE_DEVICE_PROPERTIES
                self._check(lib.zeDeviceGetProperties(dev, ctypes.byref(props)), "zeDeviceGetProperties")
                if props.type == self.ZE_DEVICE_TYPE_GPU:
                    self.device, self._driver = dev, drv
                    break
            if self.device is not None:
                break
        if self.device is None:
            raise OSError("Level Zero enumerated no GPU device")

        cdesc = self._ContextDesc()
        cdesc.stype = self.ZE_STYPE_CONTEXT_DESC
        self.context = ctypes.c_void_p()
        self._check(lib.zeContextCreate(self._driver, ctypes.byref(cdesc), ctypes.byref(self.context)), "zeContextCreate")
        self.desc = f"zeMemAllocDevice(Level Zero, flags=0x{ZE_ALLOC_FLAGS:x})"
        if contig:
            self.desc = "zePhysicalMemCreate+zeVirtualMemMap(Level Zero, physically contiguous)"
        # self-check: 1MB alloc/free
        self.free(self.alloc(MB))

    def _check(self, rc: int, what: str):
        if rc != self.ZE_SUCCESS:
            raise OSError(f"{what} failed rc=0x{rc & 0xFFFFFFFF:x}")

    def _oom_or_raise(self, rc: int, what: str):
        if rc in (self.ZE_OOM_DEVICE, self.ZE_OOM_HOST):
            raise RawOOM(f"{self.desc}: KMD out of memory ({what}, rc=0x{rc & 0xFFFFFFFF:x})")
        self._check(rc, what)

    def alloc(self, nbytes: int) -> int:
        if self.contig:
            return self._alloc_phys(nbytes)
        desc = self._MemAllocDesc()
        desc.stype = self.ZE_STYPE_DEVICE_MEM_ALLOC_DESC
        desc.flags = ZE_ALLOC_FLAGS
        desc.ordinal = 0
        ptr = ctypes.c_void_p()
        rc = self._lib.zeMemAllocDevice(
            self.context, ctypes.byref(desc), ctypes.c_size_t(nbytes),
            ctypes.c_size_t(4096), self.device, ctypes.byref(ptr),
        )
        if rc != self.ZE_SUCCESS:
            kind = "KMD out of memory" if rc in (self.ZE_OOM_DEVICE, self.ZE_OOM_HOST) else "rc"
            raise RawOOM(f"{self.desc}: {kind} (rc=0x{rc & 0xFFFFFFFF:x})")
        return ptr.value

    def free(self, ptr: int):
        if self.contig:
            self._free_phys(ptr)
            return
        rc = self._lib.zeMemFree(self.context, ctypes.c_void_p(ptr))
        if rc != self.ZE_SUCCESS:
            print(f"  ⚠️ {self.desc} free rc=0x{rc & 0xFFFFFFFF:x}")

    def sync(self):
        rc = self._lib.zeDeviceSynchronize(self.device)
        if rc != self.ZE_SUCCESS:
            print(f"  ⚠️ {self.desc} sync rc=0x{rc & 0xFFFFFFFF:x}")

    def _alloc_phys(self, nbytes: int):
        """Physically contiguous KMD alloc: one zePhysicalMemCreate object (single
        contiguous physical backing, no page scatter) mapped into a reserved VA range."""
        ctx, dev = self.context, self.device
        ps = ctypes.c_size_t(0)
        rc = self._lib.zeVirtualMemQueryPageSize(ctx, dev, nbytes, ctypes.byref(ps))
        self._check(rc, "zeVirtualMemQueryPageSize")
        vsize = ((nbytes + ps.value - 1) // ps.value) * ps.value
        va = ctypes.c_void_p()
        rc = self._lib.zeVirtualMemReserve(ctx, None, vsize, ctypes.byref(va))
        if rc != self.ZE_SUCCESS:
            self._oom_or_raise(rc, "zeVirtualMemReserve")
        pdesc = self._PhysicalMemDesc()
        pdesc.stype = self.ZE_STYPE_PHYSICAL_MEM_DESC
        pdesc.flags = self.ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE
        pdesc.size = vsize
        phys = ctypes.c_void_p()
        rc = self._lib.zePhysicalMemCreate(ctx, dev, ctypes.byref(pdesc), ctypes.byref(phys))
        if rc != self.ZE_SUCCESS:
            self._lib.zeVirtualMemFree(ctx, va, vsize)
            self._oom_or_raise(rc, "zePhysicalMemCreate")
        rc = self._lib.zeVirtualMemMap(ctx, va, vsize, phys, 0, self.ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE)
        if rc != self.ZE_SUCCESS:
            self._lib.zePhysicalMemDestroy(ctx, phys)
            self._lib.zeVirtualMemFree(ctx, va, vsize)
            self._oom_or_raise(rc, "zeVirtualMemMap")
        return (phys.value, va.value, vsize)

    def _free_phys(self, h):
        phys, va, vsize = h
        ctx = self.context
        self._lib.zeVirtualMemUnmap(ctx, ctypes.c_void_p(va), vsize)
        self._lib.zePhysicalMemDestroy(ctx, ctypes.c_void_p(phys))
        rc = self._lib.zeVirtualMemFree(ctx, ctypes.c_void_p(va), vsize)
        if rc != self.ZE_SUCCESS:
            print(f"  ⚠️ {self.desc} free rc=0x{rc & 0xFFFFFFFF:x}")


# ========= Backend abstraction =========
class Backend:
    NAME = "?"
    TDEVICE = "?"
    _mod = None

    def __init__(self):
        self.raw = None      # raw driver allocator or None
        self.raw_err = None  # why raw is unavailable (shown on --direct degrade)

    def available(self) -> bool:
        raise NotImplementedError

    def _try_raw(self, contig: bool = False):
        return None

    def setup(self, direct: bool, contig: bool = False):
        if not direct:
            return
        if contig and self.NAME != "xpu":
            print(f"  ℹ️ {self.NAME}: --contig is a no-op ({self.NAME} allocs are already contiguous-demanding)")
        try:
            self.raw = self._try_raw(contig)
        except Exception as e:
            self.raw, self.raw_err = None, str(e)

    def device_name(self):
        try:
            return self._mod.get_device_name(0)
        except Exception:
            return self.NAME

    def mem_info(self):
        """(free, total) from the driver (KMD) view; (None, None) if unavailable."""
        try:
            return tuple(self._mod.mem_get_info())
        except Exception:
            return None, None

    def empty_cache(self):
        self._mod.empty_cache()

    def synchronize(self):
        self._mod.synchronize()

    def reserved(self):
        return self._mod.memory_reserved()

    def alloc(self, nbytes: int, fill: bool = False):
        """Torch tensor alloc (UMD caching-allocator path)."""
        t = torch.empty(nbytes // 4, dtype=torch.float32, device=self.TDEVICE)
        if fill:
            t.fill_(1.0)
        return t


class CudaLikeBackend(Backend):
    _is_hip = False

    def __init__(self):
        super().__init__()
        self._mod = torch.cuda

    def available(self):
        return self._is_hip == bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()

    def _try_raw(self, contig: bool = False):
        return RawDriver("hip" if self._is_hip else "cuda")


class CudaBackend(CudaLikeBackend):
    NAME = TDEVICE = "cuda"


class HipBackend(CudaLikeBackend):
    NAME = "hip"
    TDEVICE = "cuda"
    _is_hip = True


class XpuBackend(Backend):
    NAME = TDEVICE = "xpu"

    def __init__(self):
        super().__init__()
        self._mod = torch.xpu

    def available(self):
        return torch.xpu.is_available()

    def _try_raw(self, contig: bool = False):
        return LevelZeroDriver(contig=contig)


BACKENDS = {"cuda": CudaBackend, "hip": HipBackend, "xpu": XpuBackend}


def resolve_backend(name: str):
    hip = bool(getattr(torch.version, "hip", None))
    order = ["cuda", "hip", "xpu"] if name == "auto" else [name]
    for n in order:
        if n == "cuda" and hip:
            continue  # under ROCm, torch.cuda is actually hip
        if n == "hip" and not hip:
            continue
        try:
            bk = BACKENDS[n]()
            if bk.available():
                return bk
        except Exception:
            pass
    return None


# ========= Measurement layers: UMD pool vs KMD direct =========
class PoolLayer:
    """UMD layer: torch tensors through the caching allocator.

    Gauge is reserved MB (counts up), sync is torch-side, fill is supported, and
    tensors free when the caller drops the reference.
    """

    layer_name = "UMD pool"

    def __init__(self, bk: Backend, degraded: bool = False):
        self.bk = bk
        self.label = (
            "KMD direct (empty_cache through UMD pool)"
            if degraded
            else "UMD cached pool (torch caching allocator)"
        )

    @property
    def name(self):
        return self.label

    def alloc(self, nbytes: int, fill: bool = False):
        return self.bk.alloc(nbytes, fill=fill)

    def free(self, t):
        pass  # tensor dies when the caller drops the reference

    def sync(self):
        self.bk.synchronize()

    def release(self, ts):
        ts.clear()
        gc.collect()
        self.sync()

    def gauge_mb(self):
        return self.bk.reserved() / MB

    def cold_alloc(self, nbytes: int) -> float:
        """One cold alloc (ms): empty the cache first to force a KMD hit."""
        self.bk.empty_cache()
        t0 = time.perf_counter()
        t = self.bk.alloc(nbytes)
        dt = (time.perf_counter() - t0) * 1e3
        del t
        return dt

    def hot_alloc_avg(self, nbytes: int, rounds: int) -> float:
        """Mean hot alloc (ms): every call is served by the cache pool."""
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            t = self.bk.alloc(nbytes)
            times.append((time.perf_counter() - t0) * 1e3)
            del t
        return sum(times) / len(times)


class KmdLayer:
    """KMD direct layer: every alloc/free is a driver call (requires bk.raw).

    Gauge is driver free MB (counts down), sync is driver-side, and fill is
    unsupported (no tensor wraps the raw pointer).
    """

    layer_name = "KMD direct"

    def __init__(self, bk: Backend):
        self.bk, self.raw = bk, bk.raw

    @property
    def name(self):
        return f"KMD direct ({self.raw.desc})"

    def alloc(self, nbytes: int, fill: bool = False):
        return self.raw.alloc(nbytes)

    def free(self, h):
        self.raw.free(h)

    def sync(self):
        self.raw.sync()

    def release(self, hs):
        for h in hs:
            self.raw.free(h)
        hs.clear()
        self.sync()

    def gauge_mb(self):
        f, _ = self.bk.mem_info()
        return (f / MB) if f is not None else None

    def cold_alloc(self, nbytes: int) -> float:
        """One cold alloc (ms): straight into KMD, freed right after."""
        t0 = time.perf_counter()
        p = self.raw.alloc(nbytes)
        dt = (time.perf_counter() - t0) * 1e3
        self.raw.free(p)
        return dt

    def hot_alloc_avg(self, nbytes: int, rounds: int) -> float:
        """Mean hot alloc (ms): KMD is hit on every call, no pool in between."""
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            p = self.raw.alloc(nbytes)
            times.append((time.perf_counter() - t0) * 1e3)
            self.raw.free(p)
        return sum(times) / len(times)


def make_layer(bk: Backend, direct: bool) -> "PoolLayer | KmdLayer":
    """--direct picks the KMD layer; without a raw driver API it degrades to the pool."""
    if not direct:
        return PoolLayer(bk)
    if bk.raw is not None:
        return KmdLayer(bk)
    print(f"  ℹ️ {bk.NAME}: no direct driver API ({bk.raw_err}); --direct degrades to per-alloc empty_cache")
    return PoolLayer(bk, degraded=True)


# ========= Helpers =========
def is_oom(e: BaseException) -> bool:
    if isinstance(e, RawOOM):
        return True
    out_cls = getattr(torch, "OutOfMemoryError", None)
    if out_cls is not None and isinstance(e, out_cls):
        return True
    return "out of memory" in str(e).lower()


def section(title: str):
    print(f"\n── {title} " + "─" * max(4, WIDTH - len(title) - 4))


def write_bandwidth(bk: Backend, nbytes: int) -> float:
    """Torch-path write bandwidth GB/s (all modes, measured via tensor fill)."""
    bk.empty_cache()
    t = bk.alloc(nbytes)
    bk.synchronize()
    t0 = time.perf_counter()
    t.fill_(1.0)
    bk.synchronize()
    dt = time.perf_counter() - t0
    del t
    return nbytes / GB / dt


def bench_small(L: "PoolLayer | KmdLayer", n: int = 500, size_mb: int = 1, rounds: int = 3):
    """Block-alloc bench (warm-up + multi-round steady state); returns (ok, steady ms = per-round min, median ms).

    Kills two biases: UMD pool cold-vs-warm (warm-up first) and KMD direct first-touch
    page/bookkeeping cost (per-round min). The remaining raw-vs-fragmented gap is page
    history; the severe-frag test's churned-empty control splits it into topology
    overhead vs history heat.
    """
    warm = []
    for _ in range(n):
        try:
            warm.append(L.alloc(size_mb * MB))
        except Exception:
            break
    L.release(warm)

    ok_best, avgs = 0, []
    for _ in range(rounds):
        ok, hs = 0, []
        t0 = time.perf_counter()
        for _ in range(n):
            try:
                hs.append(L.alloc(size_mb * MB))
                ok += 1
            except Exception as e:
                if not is_oom(e):
                    raise
                break
        avgs.append((time.perf_counter() - t0) / max(ok, 1) * 1e3)
        ok_best = max(ok_best, ok)
        L.release(hs)
    return ok_best, min(avgs), sorted(avgs)[len(avgs) // 2]


# ========= Test 1: baseline alloc/free performance =========
def bench_alloc_perf(bk: Backend, L: "PoolLayer | KmdLayer", args):
    rows = []
    for size_mb in SIZES_MB:
        if size_mb * MB > args.budget * 0.8:
            continue
        nbytes = size_mb * MB
        cold = L.cold_alloc(nbytes)
        hot = L.hot_alloc_avg(nbytes, args.rounds)
        bw = write_bandwidth(bk, nbytes)
        rows.append({"size_mb": size_mb, "cold_ms": round(cold, 4), "hot_ms": round(hot, 4), "bw_gbs": round(bw, 1)})
    for r in rows:
        print(
            f"  {r['size_mb']:>5}MB   cold {r['cold_ms']:8.4f}ms   hot {r['hot_ms']:8.4f}ms   {r['bw_gbs']:6.1f} GB/s"
        )
    if not rows:
        print("  (sizes exceed budget, nothing measured)")
    args.report["baseline"] = {"layer": L.name, "rows": rows}


# ========= Test 2: anti-fragmentation (one variant per measurement layer) =========
def test_anti_fragmentation_pool(bk: Backend, L: PoolLayer, args):
    """Pool semantics: segment reuse after a whole-segment free (reserved-MB gauge),
    and reserved growth when a whole-segment request must merge around pinned blocks."""
    rep = args.report.setdefault("anti_frag", {"layer": L.layer_name})
    seg = int(args.budget * 0.5 / MB) // 6 * 6
    if seg < 48:
        raise RuntimeError("device memory too small for the fragmentation test")
    blk = seg // 6
    parts = []

    def flush():
        for key, msg in parts:
            print(f"  {key:<7}: {msg}")

    # 1) whole segment in, then out -> the pool keeps the freed segment cached
    big = L.alloc(seg * MB, fill=True)
    del big
    gc.collect()
    L.sync()
    g0 = L.gauge_mb()

    # 2) carve 6 blocks: should be served from the cached segment (reuse -> ~0 growth)
    blocks = [L.alloc(blk * MB, fill=True) for _ in range(6)]
    growth = L.gauge_mb() - g0
    parts.append(("carve", f"6x{blk}MB, reserved +{growth:.1f}MB"))
    rep["segment_reuse_growth_mb"] = round(growth, 1)

    # 3) alternate frees -> 3 holes pinned by live blocks
    holes = {1, 3, 5}
    blocks = [t for i, t in enumerate(blocks) if i not in holes]
    gc.collect()
    L.sync()

    # 4) 3-block contiguous request: the holes are 1-block each, so a fresh region is
    #    required (h3 stays alive to pin it through the merge check below)
    rep["hole3_mb"] = blk * 3
    h3 = None
    try:
        h3 = L.alloc(blk * 3 * MB)
        parts.append(("hole3", f"{blk * 3}MB ok"))
        rep["hole3"] = "ok"
    except Exception as e:
        if not is_oom(e):
            raise
        parts.append(("hole3", f"{blk * 3}MB oom"))
        rep["hole3"] = "oom"

    # 5) release survivors, then one whole-segment request (merge / reclaim)
    L.release(blocks)
    r1 = bk.reserved()
    merged = L.alloc(seg * MB)
    growth2 = bk.reserved() / MB - r1
    parts.append(("merge", f"reserved +{growth2:.1f}MB"))
    rep["merge_growth_mb"] = round(growth2, 1)
    del merged
    del h3
    bk.empty_cache()
    flush()


def test_anti_fragmentation_kmd(L: KmdLayer, args):
    """KMD semantics: page-level replan after a whole-segment free, service of a
    3-block request over non-contiguous holes, and whole-segment reclaim."""
    rep = args.report.setdefault("anti_frag", {"layer": L.layer_name})
    seg = int(args.budget * 0.5 / MB) // 6 * 6
    if seg < 48:
        raise RuntimeError("device memory too small for the fragmentation test")
    blk = seg // 6
    parts = []

    def flush():
        for key, msg in parts:
            print(f"  {key:<7}: {msg}")

    # 1) whole segment in, then out -> one whole free region
    big = L.alloc(seg * MB)
    L.free(big)
    L.sync()
    g0 = L.gauge_mb()

    # 2) carve 6 blocks in the freed region (KMD replans pages)
    blocks = []
    try:
        for _ in range(6):
            blocks.append(L.alloc(blk * MB))
    except RawOOM as e:
        L.release(blocks)
        parts.append(("replan", f"fail ({e})"))
        rep["replan"] = "fail"
        flush()
        return
    g1 = L.gauge_mb()
    used = (g0 - g1) if (g0 is not None and g1 is not None) else None
    note = f" ({used:.0f}MB KMD_free consumed, 6 blocks resident)" if used is not None else ""
    parts.append(("replan", f"ok{note}"))
    rep["replan"] = "ok"

    # 3) alternate frees -> 3 holes pinned by live blocks
    holes = {1, 3, 5}
    for i in holes:
        L.free(blocks[i])

    # 4) 3-block contiguous request: KMD page-level scatter usually serves it
    #    from non-contiguous pages
    rep["hole3_mb"] = blk * 3
    try:
        h3 = L.alloc(blk * 3 * MB)
        parts.append(("hole3", f"{blk * 3}MB ok"))
        rep["hole3"] = "ok"
        L.free(h3)
    except Exception as e:
        if not is_oom(e):
            raise
        parts.append(("hole3", f"{blk * 3}MB oom"))
        rep["hole3"] = "oom"

    # 5) return everything, then one whole-segment request (reclaim)
    for i in (0, 2, 4):
        L.free(blocks[i])
    try:
        p = L.alloc(seg * MB)
        parts.append(("reclaim", f"{seg}MB ok"))
        rep["whole_segment"] = "ok"
        L.free(p)
    except RawOOM as e:
        parts.append(("reclaim", f"{seg}MB fail ({e})"))
        rep["whole_segment"] = "oom"
    flush()


# ========= Test 3: alloc performance under severe fragmentation =========
def test_severe_fragmentation(bk: Backend, L: "PoolLayer | KmdLayer", args):
    """Layer-agnostic: fill to the budget, free a random half, then measure ladder /
    small-block service; the churned-empty control splits topology vs history heat."""
    rep = args.report.setdefault("severe_frag", {"layer": L.layer_name})
    rng = random.Random(args.seed)
    bs_n = min(500, max(10, 1600 // args.bench_mb))

    # Phase 0: pristine (pre-fragmentation) steady-state baseline
    _, steady0, _ = bench_small(L, n=bs_n, size_mb=args.bench_mb)
    bk.empty_cache()

    # Phase 1: random-size fill up to the budget (OOM ends the fill; --ratio 1.0
    # consumes all free memory -> real fragmentation with no large contiguous region)
    handles, sizes, t_allocs = [], [], []
    target, filled = args.budget, 0
    while filled < target:
        mb = rng.choice(FILL_SIZES_MB)
        t0 = time.perf_counter()
        try:
            handles.append(L.alloc(mb * MB, fill=True))
        except Exception as e:
            if not is_oom(e):
                raise
            break
        t_allocs.append(time.perf_counter() - t0)
        sizes.append(mb)
        filled += mb
    rep["fill_blocks"] = len(sizes)
    rep["fill_mb"] = filled
    if t_allocs:
        rep["fill_alloc_avg_ms"] = round(sum(t_allocs) / len(t_allocs) * 1e3, 4)
    rep["plan_fill_sizes"] = sizes

    # Phase 2: free a random half -> severe fragmentation
    order = list(range(len(handles)))
    rng.shuffle(order)
    to_free = set(order[: len(order) // 2])
    freed_mb = sum(sizes[i] for i in to_free)
    freed = [h for i, h in enumerate(handles) if i in to_free]
    handles = [h for i, h in enumerate(handles) if i not in to_free]
    sizes = [s for i, s in enumerate(sizes) if i not in to_free]
    L.release(freed)
    rep["plan_freed_indices"] = sorted(to_free)

    # Phase 3: growing contiguous allocs on the fragmented layer
    max_ok, first_fail = 0, None
    for mb in [s for s in LADDER_MB if s * MB <= args.budget]:
        try:
            h = L.alloc(mb * MB)
        except Exception as e:
            if not is_oom(e):
                raise
            first_fail = mb
            break
        max_ok = mb
        L.free(h)
        h = None

    # Phase 4: small-block service on the fragmented layer
    ok4, steady4, _ = bench_small(L, n=bs_n, size_mb=args.bench_mb)
    rep["small_bench_mb"] = args.bench_mb
    rep["small_ok"] = ok4
    rep["small_steady_ms"] = round(steady4, 4)

    # Phase 4b: churned-empty control -- release the survivors (page history stays hot,
    # topology returns to contiguous); splits topology overhead vs history heat
    L.release(handles)
    _, steady5b, _ = bench_small(L, n=bs_n, size_mb=args.bench_mb)
    topo = (steady4 / steady5b) if steady5b else None
    hist = (steady5b / steady0) if steady0 else None
    rep["small_steady_ms_pristine"] = round(steady0, 4)
    rep["small_steady_ms_churn"] = round(steady5b, 4)
    rep["small_frag_overhead"] = round(topo, 2) if topo is not None else None
    rep["small_history_ratio"] = round(hist, 2) if hist is not None else None

    # Phase 5: recovery -- retry the previously OOM size after cleanup
    recovered = None
    if first_fail:
        try:
            h = L.alloc(first_fail * MB)
            recovered = True
            L.free(h)
        except Exception as e:
            if not is_oom(e):
                raise
            recovered = False
    rep["recovered"] = recovered
    rep["ladder_max_ok_mb"] = max_ok
    rep["ladder_first_fail_mb"] = first_fail

    ladder_txt = f"{max_ok}MB ok" + (f", OOM from {first_fail}MB" if first_fail else "")
    print(f"  {'fill':<7}: {rep['fill_blocks']} blocks / {filled}MB  (freed {len(to_free)} / {freed_mb}MB)")
    print(f"  {'ladder':<7}: {ladder_txt}")
    print(f"  {'small':<7}: {ok4}/{bs_n} blocks @ {args.bench_mb}MB -> {steady4:.4f}ms")
    print(f"  {'verdict':<7}: frag {topo:.2f}x ({steady4:.4f} vs {steady5b:.4f}ms) | history {hist:.2f}x")
    if recovered is False:
        print(f"  ⚠️ {first_fail}MB still fails after cleanup")
    bk.empty_cache()


# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser(description="Multi-backend GPU memory allocator benchmark (UMD pool / KMD direct)")
    p.add_argument("--backend", choices=["auto", "cuda", "hip", "xpu"], default="auto", help="compute backend")
    p.add_argument("--direct", action="store_true", help="bypass the UMD buffer pool, one KMD alloc per call")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ratio", type=float, default=0.35, help="test budget as a fraction of driver free memory; 1.0 fills all free memory"
    )
    p.add_argument("--rounds", type=int, default=20, help="baseline hot-alloc rounds")
    p.add_argument("--bench-mb", type=int, default=1, help="small-block bench size (MB): 1=small, 32=mid (exposes frag overhead better)")
    p.add_argument("--json", default=None, help="perf report JSON output path (for A/B comparison)")
    p.add_argument(
        "--ze-uncached",
        action="store_true",
        help="xpu --direct: use Level Zero BIAS_UNCACHED alloc (GPU UC caching, not a pool bypass)",
    )
    p.add_argument(
        "--contig",
        action="store_true",
        help="xpu --direct: KMD allocs via zePhysicalMemCreate (physically contiguous, no page scatter); no-op for cuda/hip",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.report = {}
    if args.ze_uncached:
        global ZE_ALLOC_FLAGS
        ZE_ALLOC_FLAGS = 0x2  # ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED

    bk = resolve_backend(args.backend)
    if bk is None:
        raise SystemExit(
            f"❌ backend {args.backend} unavailable (auto probed cuda/hip/xpu, all failed; "
            f"a sandbox may block /dev/dri — run on the host terminal)"
        )

    if args.contig and not args.direct:
        raise SystemExit("❌ --contig requires --direct")

    bk.setup(direct=args.direct, contig=args.contig)
    free, total = bk.mem_info()
    budget = int((free or 0) * args.ratio)
    if budget <= 0:
        raise SystemExit("❌ no usable budget: driver free memory unknown or 0 (mem_get_info failed)")
    args.budget = budget
    L = make_layer(bk, args.direct)
    print("═" * WIDTH)
    print(" GPU Memory Allocator Benchmark")
    print("═" * WIDTH)
    print(f"  {'torch':<8}: {torch.__version__}")
    print(f"  {'backend':<8}: {bk.NAME} ({bk.device_name()})")
    if free is not None:
        print(f"  {'memory':<8}: {free / GB:.1f} / {total / GB:.1f} GB free")
    print(f"  {'layer':<8}: {L.name}")
    print(f"  {'budget':<8}: {budget / GB:.2f} GB   seed {args.seed}")
    args.report["meta"] = {
        "torch": torch.__version__,
        "backend": bk.NAME,
        "device": bk.device_name(),
        "layer": L.name,
        "direct": args.direct,
        "contig": args.contig,
        "seed": args.seed,
        "budget_gb": round(budget / GB, 2),
        "env": {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)},
    }

    section("Baseline alloc/free")
    bench_alloc_perf(bk, L, args)
    section("Anti-fragmentation")
    if isinstance(L, KmdLayer):
        test_anti_fragmentation_kmd(L, args)
    else:
        test_anti_fragmentation_pool(bk, L, args)
    section("Severe fragmentation")
    test_severe_fragmentation(bk, L, args)

    bk.empty_cache()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(args.report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ JSON report -> {args.json}")
    else:
        print("\n✅ done")


if __name__ == "__main__":
    main()
