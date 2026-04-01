import os
import pickle

SRC = "./sunrgbd/data/pickle_data/sunrgbd_rgb_det_val.pickle"
DST = "./sunrgbd/data/pickle_data/sunrgbd_rgb_det_val.numpy126.pickle"

_REMAP_PREFIX = [
    ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"),
    ("numpy._core.multiarray",        "numpy.core.multiarray"),
    ("numpy._core.numeric",           "numpy.core.numeric"),
    ("numpy._core.umath",             "numpy.core.umath"),
    ("numpy._core",                   "numpy.core"),
]

class Numpy20CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for src, dst in _REMAP_PREFIX:
            if module == src or module.startswith(src + "."):
                module = module.replace(src, dst, 1)
                break
        return super().find_class(module, name)

def load_compat(path: str):
    with open(path, "rb") as f:
        return Numpy20CompatUnpickler(f).load()

def dump_plain(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        # protocol=4 通常对兼容性更友好
        pickle.dump(obj, f, protocol=4)

if __name__ == "__main__":
    obj = load_compat(SRC)
    dump_plain(obj, DST)
    print("Converted OK:")
    print("  SRC:", SRC)
    print("  DST:", DST)

    # quick sanity check: ensure DST loads with vanilla pickle.load
    with open(DST, "rb") as f:
        obj2 = pickle.load(f)
    print("DST vanilla load OK. keys:", list(obj2.keys()))