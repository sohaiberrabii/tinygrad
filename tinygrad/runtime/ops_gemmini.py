# a python uops emulator for systolic arrays
from typing import Tuple, List, Optional, Any, Dict
import pickle, time, struct
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv
from tinygrad.device import Compiled
from tinygrad.ops import BinaryOps, TernaryOps, exec_alu, Ops, GroupOp
from tinygrad.renderer import TensorCore
from tinygrad.runtime.ops_python import PythonCompiler, PythonRenderer, PythonAllocator
from samodel import MeshWithDelays

def load(m, i):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class GemminiProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[Ops, Optional[DType], List[int], Any]] = pickle.loads(lib)
    self.sa = MeshWithDelays(2, 2, 1, 1)
  def __call__(self, *bufs, **_):
    st = time.perf_counter()
    ul: Dict[int, Any] = {}
    dl: Dict[int, DType] = {}
    pbufs: List[memoryview] = list(bufs)
    i = 0
    loop_ends: Dict[int, int] = {}
    while i < len(self.uops):
      uop, dtype, idp, arg = self.uops[i]
      void_ops = {Ops.STORE, Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF}
      if uop is Ops.DEFINE_ACC: idp = [idp[0]]
      inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
      dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]

      if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)

      if uop is Ops.STORE:
        assert len(inp) == 2 and len(inp[0]) == 2
        for j, val in enumerate(inp[1]):
          store(inp[0][0], inp[0][1] + j, val)
        i += 1
        continue

      if uop is Ops.ENDRANGE:
        loop_ends[idp[0]] = i
        i = idp[0]
        continue

      assert dtype is not None, f"{uop} is missing a dtype"
      dl[i] = dtype

      if uop is Ops.LOAD:
        assert len(inp) == 1 and len(inp[0]) == 2
        ul[i] = [load(inp[0][0], inp[0][1] + j) for j in range(dtype.count)]

      elif uop is Ops.DEFINE_GLOBAL:
        assert dtype.fmt is not None
        ul[i] = pbufs.pop(0).cast(dtype.fmt)

      elif uop is Ops.CONST: ul[i] = arg

      elif uop is Ops.INDEX:
        assert not isinstance(dtp[0], ImageDType)
        ul[i] = inp

      elif uop is Ops.CAST and isinstance(dtype, PtrDType):
        ul[i] = inp[0]

      elif uop is Ops.RANGE:
        if i not in ul: ul[i] = inp[0]
        else:
          ul[i] += 1
          if ul[i] == inp[1]:
            del ul[i]
            i = loop_ends[i] + 1
            continue

      elif uop is Ops.VECTORIZE: ul[i] = inp

      elif uop in {Ops.CAST, Ops.BITCAST}:
        assert dtp[0].fmt and dtype.fmt
        pack_format, unpack_format = dtp[0].fmt, dtype.fmt
        if uop is Ops.BITCAST: ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
        else: ul[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in inp[0]]

      elif uop is Ops.ASSIGN:
        for j in range(len(inp[0])): inp[0][j] = inp[1][j]
        ul[i] = inp[0]

      elif uop is Ops.GEP:
        assert len(arg) == 1 and len(inp) == 1
        ul[i] = inp[0][arg[0]]

      elif uop is Ops.WMMA:
        self.sa.preload_weights(inp[1])
        ul[i] = self.sa.preloaded_matmul(inp[0], inp[2]).reshape(-1).tolist()

      elif uop in GroupOp.ALU:
        assert all_same([dtype] + dtp) or uop in {BinaryOps.CMPNE, BinaryOps.CMPLT, TernaryOps.WHERE}, f"dtype mismatch on {uop}"
        assert dtype.count == 1 or all_same([len(x) for x in inp]), f"operands {inp} don't match"
        ul[i] = exec_alu(uop, dtype, inp)
      else:
        raise NotImplementedError(f"Unsupported uop: {uop}")
      assert i in ul, (uop, dtype, idp, arg)
      i += 1
    return time.perf_counter() - st

class GemminiRenderer(PythonRenderer):
  device = "GEMMINI"
  tensor_cores = [TensorCore( dims=(sz, sz, sz), threads=[], reduce_axes=[(0, sz)],
    upcast_axes=([(2,sz), (0,sz)],[(0, sz), (1,sz)],[(2,sz),(1,sz)]), dtype_in=dt, dtype_out=dt) for dt, sz in [(dtypes.float, 2)]]
  has_local=False

class GemminiDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), GemminiRenderer(), PythonCompiler(), GemminiProgram)
