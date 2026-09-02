try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    cp.zeros(1)
    CUPY_DISPONIBLE = True
except Exception:
    cp = None
    cpsp = None

    CUPY_DISPONIBLE = False
