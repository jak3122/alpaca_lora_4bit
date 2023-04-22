from sentencepiece import SentencePieceProcessor # type: ignore
import json, struct, os, re, zipfile, pickle, itertools, sys, enum, threading, concurrent.futures, argparse
from pathlib import Path
import numpy as np
from collections import namedtuple
from typing import Optional, Callable, Type, Any, Iterable, IO, Sequence, Union, TypeVar
from dataclasses import dataclass

NDArray = np.ndarray[Any, Any]

DataType = enum.Enum('DataType', ['F16', 'F32', 'I32', 'BF16', 'Q4_1'])

DATA_TYPE_TO_FTYPE: dict[DataType, int] = {
    DataType.F32: 0,
    DataType.F16: 1,
    DataType.Q4_1: 3,
}
DATA_TYPE_TO_NUMPY: dict[DataType, Type[np.generic]] = {
    DataType.F16: np.float16,
    DataType.F32: np.float32,
    DataType.I32: np.int32,
}

def make_tensors_list() -> list[str]:
    ret = [
        'tok_embeddings.weight',
        'norm.weight',
        'output.weight',
    ]
    for i in range(80): # maximum number of layer
        ret += [
            f'layers.{i}.attention.wq.weight',
            f'layers.{i}.attention.wk.weight',
            f'layers.{i}.attention.wv.weight',
            f'layers.{i}.attention.wo.weight',
            f'layers.{i}.attention_norm.weight',
            f'layers.{i}.feed_forward.w1.weight',
            f'layers.{i}.feed_forward.w2.weight',
            f'layers.{i}.feed_forward.w3.weight',
            f'layers.{i}.atttention_norm.weight',
            f'layers.{i}.ffn_norm.weight',
        ]
    return ret
TENSORS_LIST = make_tensors_list()
TENSORS_SET = set(TENSORS_LIST)

def always_want_f32(name: str) -> bool:
    return (name.endswith('.attention_norm.weight') or
            name.endswith('.ffn_norm.weight') or
            name == 'norm.weight')

@dataclass
class Params:
    n_vocab: int
    n_embd: int
    n_mult: int
    n_head: int
    n_layer: int
    file_type: int

    @staticmethod
    def guessed(model: 'LazyModel') -> 'Params':
        n_vocab, n_embd = model["tok_embeddings.weight"].shape

        return Params(
            n_vocab = n_vocab,
            n_embd = n_embd,
            n_mult = 256,
            n_head = n_embd // 128,
            n_layer = next(i for i in itertools.count() if f"layers.{i}.attention.wq.weight" not in model),
            file_type = Params.guess_file_type(model),
        )

    @staticmethod
    def guess_file_type(model: 'LazyModel') -> int:
        name_to_type: dict[str, DataType] = {}
        for name, tensor in model.items():
            if always_want_f32(name):
                assert tensor.data_type == DataType.F32, name
            else:
                name_to_type[name] = tensor.data_type

        types = set(name_to_type.values())
        if len(types) == 1:
            # All the same type.
            return DATA_TYPE_TO_FTYPE[next(iter(types))]

        # Could it be type 4?
        if all(data_type == (DataType.F16 if name in ("tok_embeddings.weight", "output.weight")
                                                 else DataType.Q4_1)
               for (name, data_type) in name_to_type.items()):
            return 4
        raise Exception(f"Unknown data types: {name_to_type}")

class Vocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            text: bytes
            if tokenizer.is_unknown(i):
                text = " \u2047 ".encode("utf-8")
            elif tokenizer.is_control(i):
                text = b""
            elif tokenizer.is_byte(i):
                piece = tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    raise Exception(f"Invalid token: {piece}")
                byte_value = int(piece[3:-1], 16)
                text = struct.pack("B", byte_value)
            else:
                text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            score: float = tokenizer.get_score(i)
            yield text, score

    def added_tokens(self) -> Iterable[tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

def dequantize_q4(qvalues_pack32: NDArray, scales: NDArray, addends: NDArray) -> NDArray:
    # First reinterpret each row from a list of int32s containing 8 values each
    # to a list of uint8s containing 2 values each.
    qvalues_pack8 = qvalues_pack32.view(np.uint8)

    # Then split out the two values per int8 (which requires an actual
    # conversion because numpy doesn't natively support int4s).
    qvalues = np.zeros([qvalues_pack8.shape[0], qvalues_pack8.shape[1] * 2], dtype=np.uint8)
    qvalues[:, 0::2] = qvalues_pack8 & 0xf
    qvalues[:, 1::2] = qvalues_pack8 >> 4

    assert addends.shape == scales.shape
    assert qvalues.shape[0] == scales.shape[0]
    assert qvalues.shape[1] % scales.shape[1] == 0
    repeat_count = qvalues.shape[1] // scales.shape[1]
    scales = scales[:, :, np.newaxis]
    addends = addends[:, :, np.newaxis]
    # Reshape so that the below computation broadcasts over scales and addends:
    qvalues.shape = (qvalues.shape[0], scales.shape[1], int(repeat_count))
    # And do the actual 'value = scale * qvalue + addend' computation.
    values = scales * qvalues
    values += addends
    values.shape = (values.shape[0], values.shape[1] * values.shape[2])
    return values

class UnquantizedTensor:
    def __init__(self, ndarray: NDArray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
    def astype(self, dtype: Type[np.generic]) -> 'UnquantizedTensor':
        return UnquantizedTensor(self.ndarray.astype(dtype))
    def ggml_ndarray(self) -> NDArray:
        return self.ndarray

def load_unquantized(lazy_tensor: 'LazyTensor', expected_dtype: Optional[Type[np.generic]] = None) -> NDArray:
    tensor = lazy_tensor.load()
    assert isinstance(tensor, UnquantizedTensor)
    if expected_dtype is not None:
        assert tensor.ndarray.dtype == expected_dtype, (tensor.ndarray.dtype, expected_dtype)
    return tensor.ndarray

class QuantizedTensor:
    def __init__(self, model: 'LazyModel', namebase: str, permute_n_head : Optional[int] = None) -> None:
        qweight = load_unquantized(model[f"{namebase}.qweight"], np.int32)
        scales = load_unquantized(model[f"{namebase}.scales"], np.float32)

        bias = model.get(f"{namebase}.bias")
        if bias is not None:
            # Q4_1 does not support bias; good thing the bias is always all zeros.
            assert not np.any(load_unquantized(bias))

        if f"{namebase}.zeros" in model:
            zeros = load_unquantized(model[f"{namebase}.zeros"], np.float32)
        else:
            qzeros = load_unquantized(model[f"{namebase}.qzeros"], np.int32)
            assert qzeros.dtype == np.int32
            zeros = dequantize_q4(qzeros, scales, scales)
            assert zeros.dtype == np.float32
        assert zeros.shape == scales.shape

        # Output is transposed compared to the input, and addends have their sign flipped.
        # Scales and zeros similarly must be transposed but only for newer
        # versions of GPTQ-for-LLaMa; the older versions can be identified by
        # having shape (n_embd, 1).
        qweight = qweight.T
        if scales.shape[1] != 1:
            scales = scales.T
            zeros = zeros.T

        # Output also has signs flipped for the addends.
        self.qweight = qweight
        self.scales = scales
        self.addends = -zeros

        self.shape = [self.qweight.shape[0], self.qweight.shape[1] * 8]
        self.permute_n_head = permute_n_head

    def inspect(self, row: int, col: int) -> None:
        '''For debugging.'''
        if self.permute_n_head is not None:
            permute_group_size = self.qweight.shape[0] // self.permute_n_head
            row_pg = row // permute_group_size
            row_pgoff = row % permute_group_size
            row_pgoff = (row_pgoff // 2) + (permute_group_size // 2) * (row_pgoff & 1)
            row = row_pg * permute_group_size + row_pgoff

        qweight = (self.qweight[row, col // 8] >> (4 * (col & 7))) & 0xf
        group = int(col // self.groupsize())
        scale = self.scales[row, group]
        addend = self.addends[row, group]
        with np.printoptions(precision=None, suppress=True):
            print(f'scale:{scale} addend:{addend} qweight:{qweight}')
            print('possible values:', np.arange(16) * scale + addend)
            print('actual value:', qweight * scale + addend)

    def astype(self, dtype: Type[np.generic]) -> UnquantizedTensor:
        '''Also for debugging.'''
        dequantized = dequantize_q4(np.ascontiguousarray(self.qweight), self.scales, self.addends)
        if self.permute_n_head is not None:
            old = dequantized
            dequantized = permute(dequantized, self.permute_n_head)
        return UnquantizedTensor(dequantized).astype(dtype)

    def groupsize(self) -> int:
        assert self.addends.shape == self.scales.shape
        assert self.shape[1] % self.scales.shape[1] == 0
        return self.shape[1] // self.scales.shape[1]

    def regroup(self, new_groupsize: int = 32) -> None:
        # Old versions of GPTQ-for-LLaMa shared scales and addends between all the
        # columns in a row.  Newer versions share them between every set of N
        # columns in a row, where N is the `groupsize` parameter, usually 128.  The
        # output format shares them between every set of 32 columns.  To handle
        # this, duplicate scales and addends for every smaller group.
        # (In the above, 'row' and 'column' are in the sense of the output.)
        old_groupsize = self.groupsize()
        assert old_groupsize >= new_groupsize and old_groupsize % new_groupsize == 0, old_groupsize
        self.addends = self.addends.repeat(old_groupsize // new_groupsize, axis=1)
        self.scales = self.scales.repeat(old_groupsize // new_groupsize, axis=1)

    def ggml_ndarray(self) -> NDArray:
        # The output format looks like this:
        # For each row:
        #   For each group of 32 columns:
        #     - addend (float32, 4 bytes)
        #     - scale (float32, 4 bytes)
        #     - weights (int4 * 32, 16 bytes)

        # Since the output format is mixed between integers and floats, we have
        # to hackily view the floats as int32s just so numpy will let us
        # concatenate them.
        self.regroup()
        addends_view = self.addends.view(dtype=np.int32)[:, :, np.newaxis]
        scales_view = self.scales.view(dtype=np.int32)[:, :, np.newaxis]

        # Split into groups of 4 columns (i.e. 32 columns of quantized data):
        grouped = self.qweight.reshape([self.qweight.shape[0], self.qweight.shape[1] // 4, 4])

        # And concatenate:
        grouped = np.concatenate([scales_view, addends_view, grouped], axis=2, casting='no')

        if self.permute_n_head is not None:
            grouped = permute(grouped, self.permute_n_head)
        return grouped


Tensor = Union[QuantizedTensor, UnquantizedTensor]

def permute(weights: NDArray, n_head: int) -> NDArray:
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))


@dataclass
class LazyTensor:
    load: Callable[[], Tensor]
    shape: Sequence[int]
    data_type: DataType

    def astype(self, data_type: DataType) -> 'LazyTensor':
        dtype = DATA_TYPE_TO_NUMPY[data_type]
        def load() -> Tensor:
            return self.load().astype(dtype)
        return LazyTensor(load, self.shape, data_type)

LazyModel = dict[str, LazyTensor]

def load_orig_llama_file(path: Path, first_model: LazyModel) -> LazyModel:
    models = []
    # Check for multi-file input
    m = re.match(r'^(.*)\.[0-9]{2}\.pth$', path.name)
    if m:
        # Load other .pth files
        base = m.group(1)
        for i in itertools.count():
            new_path = path.with_name(f"{base}.{i:02}.pth")
            try:
                models.append(first_model if new_path == path else lazy_load_torch(new_path))
            except FileNotFoundError:
                break
    else:
        models.append(first_model)

    print(f"Loaded original LLaMA model split into {len(models)} parts.")

    # Original LLaMA models have each file contain one part of each tensor.
    names = sorted(name for model in models for name in model)
    combined: LazyModel = {}
    for name in names:
        lazy_tensors: list[LazyTensor] = [model[name] for model in models]
        if len(lazy_tensors[0].shape) == 1:
            # the tensor is just duplicated in every file
            combined[name] = lazy_tensors[0]
            continue
        if (name.startswith('tok_embeddings.') or
            name.endswith('.attention.wo.weight') or
            name.endswith('.feed_forward.w2.weight')):
            # split by columns
            axis = 1
        else:
            # split by rows
            axis = 0
        concatenated_shape = list(lazy_tensors[0].shape)
        concatenated_shape[axis] = sum(tensor.shape[axis] for tensor in lazy_tensors)
        def load(axis: int = axis, lazy_tensors: list[LazyTensor] = lazy_tensors) -> UnquantizedTensor:
            ndarrays = [load_unquantized(tensor) for tensor in lazy_tensors]
            concatenated: NDArray = np.concatenate(ndarrays, axis=axis)
            return UnquantizedTensor(concatenated)
        combined[name] = LazyTensor(load, concatenated_shape, lazy_tensors[0].data_type)
    return combined

def permute_lazy(lazy_tensor: LazyTensor, n_head: int) -> LazyTensor:
    def load() -> Tensor:
        tensor = lazy_tensor.load()
        if isinstance(tensor, UnquantizedTensor):
            return UnquantizedTensor(permute(tensor.ndarray, n_head))
        else: # QuantizedTensor
            tensor.permute_n_head = n_head
            return tensor
    return LazyTensor(load, lazy_tensor.shape, lazy_tensor.data_type)

def convert_transformers_to_orig(model: LazyModel) -> LazyModel:
    out: LazyModel = {}
    out["tok_embeddings.weight"] = model["model.embed_tokens.weight"]
    out["norm.weight"] = model["model.norm.weight"]
    out["output.weight"] = model["lm_head.weight"]

    n_head = model[f"model.layers.0.self_attn.q_proj.weight"].shape[1] // 128
    for i in itertools.count():
        if f"model.layers.{i}.self_attn.q_proj.weight" not in model:
            break
        out[f"layers.{i}.attention.wq.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.q_proj.weight"], n_head)
        out[f"layers.{i}.attention.wk.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.k_proj.weight"], n_head)
        out[f"layers.{i}.attention.wv.weight"] = model[f"model.layers.{i}.self_attn.v_proj.weight"]
        out[f"layers.{i}.attention.wo.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]

        out[f"layers.{i}.feed_forward.w1.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
        out[f"layers.{i}.feed_forward.w2.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
        out[f"layers.{i}.feed_forward.w3.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]

        out[f"layers.{i}.attention_norm.weight"] = model[f"model.layers.{i}.input_layernorm.weight"]
        out[f"layers.{i}.ffn_norm.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"]
    return out

def handle_quantization(model: LazyModel) -> LazyModel:
    '''Convert a model with entries for 'foo.qweight', 'foo.scales', etc.
    (which resolve to UnquantizedTensors with the raw data) to one with entries
    for 'foo.weight' (whicih resolve to QuantizedTensors).
    '''
    out: LazyModel = {}
    for key, lazy_tensor in model.items():
        if key.endswith(".qweight"):
            namebase = key.rsplit('.', 1)[0]
            orig_name = namebase + ".weight"
            def load(model: LazyModel = model, namebase: str = namebase) -> Tensor:
                return QuantizedTensor(model, namebase)
            assert len(lazy_tensor.shape) == 2
            real_shape = (lazy_tensor.shape[1], lazy_tensor.shape[0] * 8)
            out[orig_name] = LazyTensor(load, real_shape, DataType.Q4_1)
        else:
            out[key] = lazy_tensor
    return out

def load_transformers_file(path: Path, first_model: LazyModel) -> LazyModel:
    # Check for multi-file input
    m = re.match(r'(.*)-[0-9]{5}-of-([0-9]{5})\.bin$', path.name)
    if m:
        base, count = m.group(1), int(m.group(2))
        paths = [path.with_name(f"{base}-{i:05}-of-{count:05}.bin") for i in range(1, count + 1)]
    else:
        paths = [path]

    print(f"Loaded 'transformers' model split into {len(paths)} parts.")

    # Transformers models don't split an individual tensor into multiple parts,
    # but do have multiple files with different sets of tensors.
    joined: LazyModel = {}
    for path in paths:
        for key, tensor in lazy_load_torch(path).items():
            if key in joined:
                sys.stderr.write(f"Warning: multiple .bin files contained {key!r}\n")
            joined[key] = tensor
    return convert_transformers_to_orig(handle_quantization(joined))

# Functionality that simulates `torch.load` but where individual tensors are
# only loaded into memory on demand, not all at once.
# PyTorch can't do this natively as of time of writing:
# - https://github.com/pytorch/pytorch/issues/64327
# This allows us to de-shard without multiplying RAM usage, and also
# conveniently drops the PyTorch dependency (though we still need numpy).

@dataclass
class LazyStorageKind:
    data_type: DataType
@dataclass
class LazyStorage:
    load: Callable[[int, int], NDArray]
    kind: LazyStorageKind

class LazyUnpickler(pickle.Unpickler):
    def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
        super().__init__(fp)
        self.data_base_path = data_base_path
        self.zip_file = zip_file
    def persistent_load(self, pid: Any) -> Any:
        assert pid[0] == 'storage'
        assert isinstance(pid[1], LazyStorageKind)
        data_type = pid[1].data_type
        filename_stem = pid[2]
        filename = self.data_base_path + '/' + filename_stem
        info = self.zip_file.getinfo(filename)
        def load(offset: int, elm_count: int) -> NDArray:
            dtype = DATA_TYPE_TO_NUMPY.get(data_type)
            if dtype is None:
                raise Exception("tensor stored in unsupported format")
            itemsize = dtype(0).itemsize
            fp = self.zip_file.open(info)
            fp.seek(offset * itemsize)
            size = elm_count * itemsize
            data = fp.read(size)
            assert len(data) == size
            return np.frombuffer(data, dtype)
        return LazyStorage(load=load, kind=pid[1])

    @staticmethod
    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any, requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> LazyTensor:
        assert isinstance(storage, LazyStorage)
        def load() -> UnquantizedTensor:
            elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size))
        return LazyTensor(load, size, storage.kind.data_type)

    CLASSES: dict[Any, Any] = {
        ('torch._utils', '_rebuild_tensor_v2'): lazy_rebuild_tensor_v2,
        ('torch', 'BFloat16Storage'): LazyStorageKind(DataType.BF16),
        ('torch', 'HalfStorage'): LazyStorageKind(DataType.F16),
        ('torch', 'FloatStorage'): LazyStorageKind(DataType.F32),
        ('torch', 'IntStorage'): LazyStorageKind(DataType.I32),
    }
    def find_class(self, module: str, name: str) -> Any:
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]

def lazy_load_torch(path: Path) -> LazyModel:
    zf = zipfile.ZipFile(path)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    assert len(pickle_paths) == 1, pickle_paths
    pickle_fp = zf.open(pickle_paths[0], 'r')
    unpickler = LazyUnpickler(pickle_fp,
        data_base_path = pickle_paths[0][:-4],
        zip_file = zf)
    model = unpickler.load()
    return dict(model.items())

In = TypeVar('In')
Out = TypeVar('Out')
def bounded_parallel_map(func: Callable[[In], Out], iterable: Iterable[In], concurrency: int) -> Iterable[Out]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: list[concurrent.futures.Future[Out]] = []
        items_rev = list(iterable)[::-1]
        for i in range(min(concurrency, len(items_rev))):
            futures.append(executor.submit(func, items_rev.pop()))
        while futures:
            result = futures.pop(0).result()
            if items_rev:
                futures.append(executor.submit(func, items_rev.pop()))
            yield result

def check_vocab_size(params: Params, vocab: Vocab) -> None:
    if params.n_vocab != vocab.vocab_size:
        if params.n_vocab == vocab.vocab_size_base:
            print("Ignoring added_tokens.json since model matches vocab size without it.")
            vocab.added_tokens_list = []
            vocab.vocab_size = vocab.vocab_size_base
            return
        msg = f"Vocab size mismatch (model has {params.n_vocab}, but {vocab.fname_tokenizer}"
        if vocab.fname_added_tokens is not None:
            msg += f" combined with {vocab.fname_added_tokens}"
        msg += f" has {vocab.vocab_size})."
        if vocab.vocab_size < params.n_vocab < vocab.vocab_size + 20 and vocab.fname_added_tokens is None:
            msg += f"  Most likely you are missing added_tokens.json (should be in {vocab.fname_tokenizer.parent})."
            msg += f"  Most likely added_tokens.json should not be present."
        raise Exception(msg)

class OutputFile:
    def __init__(self, fname_out: Path) -> None:
        self.fout = open(fname_out, "wb")

    def write_file_header(self, params: Params) -> None:
        values = [
            0x67676d66,  # magic: ggmf in hex
            1, # file version
            params.n_vocab,
            params.n_embd,
            params.n_mult,
            params.n_head,
            params.n_layer,
            params.n_embd // params.n_head,  # rot (obsolete)
            params.file_type,
        ]
        self.fout.write(struct.pack("i" * len(values), *values))

    def write_tensor_header(self, name: str, shape: Sequence[int], data_type: DataType) -> None:
        sname = name.encode('utf-8')
        self.fout.write(struct.pack("iii", len(shape), len(sname), DATA_TYPE_TO_FTYPE[data_type]))
        self.fout.write(struct.pack("i" * len(shape), *shape[::-1]))
        self.fout.write(sname)

    def write_vocab(self, vocab: Vocab) -> None:
        for text, score in vocab.all_tokens():
            self.fout.write(struct.pack("i", len(text)))
            self.fout.write(text)
            self.fout.write(struct.pack("f", score))

    @staticmethod
    def write_vocab_only(fname_out: Path, vocab: Vocab) -> None:
        of = OutputFile(fname_out)
        params = Params(n_vocab = vocab.vocab_size, n_embd = 0, n_mult = 0,
                        n_head = 1, n_layer = 0, file_type = 0)
        of = OutputFile(fname_out)
        of.write_file_header(params)
        of.write_vocab(vocab)
        of.fout.close()

    @staticmethod
    def write_all(fname_out: Path, params: Params, model: LazyModel, vocab: Vocab) -> None:
        check_vocab_size(params, vocab)
        of = OutputFile(fname_out)
        of.write_file_header(params)
        print(f"Writing vocab...")
        of.write_vocab(vocab)

        ndarrays = bounded_parallel_map(lambda lazy_tensor: lazy_tensor.load().ggml_ndarray(), model.values(),
                                        concurrency=8)
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            size = ' x '.join(map(str, lazy_tensor.shape))
            print(f"[{i+1}/{len(model)}] Writing tensor {name}, size {size}...")
            of.write_tensor_header(name, lazy_tensor.shape, lazy_tensor.data_type)
            ndarray.tofile(of.fout)
        of.fout.close()

def do_necessary_conversions(model: LazyModel, convert_to_float16: bool) -> LazyModel:
    out: LazyModel = model.copy()

    if model["layers.0.attention.wq.weight"].data_type == DataType.Q4_1:
        # GPTQ models may need F32->F16 for these tensors
        for name in ["tok_embeddings.weight", "output.weight"]:
            out[name] = out[name].astype(DataType.F16)
        if convert_to_float16:
            raise Exception("--convert-to-float16 is not useful with GPTQ models")

    converted = 0
    for name in out:
        if always_want_f32(name):
            out[name] = out[name].astype(DataType.F32)
        elif convert_to_float16 and out[name].data_type == DataType.F32:
            out[name] = out[name].astype(DataType.F16)
            converted += 1

    if convert_to_float16 and not converted:
        raise Exception("This model is already float16 and cannot be converted.")

    return out

def load_some_model(path: Path) -> tuple[LazyModel, Path]:
    '''Load a model of either supported format; return the model and the path where it was found.'''
    # Be extra-friendly and accept either a file or a directory:
    if path.is_dir():
        globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt"]
        files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            raise Exception(f"Can't find model in directory {path}")
        if len(files) > 1:
            raise Exception(f"Found multiple models in {path}, not sure which to pick: {files}")
        path = files[0]
    model = lazy_load_torch(path)
    if "tok_embeddings.weight" in model:
        return load_orig_llama_file(path, model), path
    else:
        return load_transformers_file(path, model), path

def filter_and_sort_tensors(model: LazyModel) -> LazyModel:
    return {name: model[name] for name in TENSORS_LIST if name in model}

def load_vocab(path: Path) -> Vocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(f"Could not find tokenizer.model in {path} or its parent; try passing --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    return Vocab(path, added_tokens_path if added_tokens_path.exists() else None)

def default_outfile(model_path: Path, params: Params) -> Path:
    namestr = {0: "f32", 1: "f16", 3: "q4_1", 4: "q4_1"}[params.file_type]
    return model_path.parent / f"ggml-model-{namestr}.bin"

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LLaMa model to a GGML compatible file")
    parser.add_argument("--vocab-only", action="store_true", help="extract only the vocab")
    parser.add_argument("--convert-to-float16", action="store_true", help="convert float32 to float16")
    parser.add_argument("--vocab-dir", type=Path, help="directory containing tokenizer.model, if separate from model file")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file, or model file itself (*.pth, *.pt, *.bin)")
    args = parser.parse_args()

    if args.vocab_only:
        vocab = load_vocab(args.vocab_dir or args.model)
        assert args.outfile, "need --outfile if using --vocab-only"
        OutputFile.write_vocab_only(args.outfile, vocab)
    else:
        model, model_path = load_some_model(args.model)
        vocab_dir = args.vocab_dir if args.vocab_dir else model_path.parent
        vocab = load_vocab(vocab_dir)
        model = filter_and_sort_tensors(model)
        model = do_necessary_conversions(model, args.convert_to_float16)
        params = Params.guessed(model)
        outfile = args.outfile or default_outfile(model_path, params)
        OutputFile.write_all(outfile, params, model, vocab)
    print(f"Wrote {outfile}")
main()
