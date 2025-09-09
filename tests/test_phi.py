import numpy as np
import torch as t

from urc.phi.transforms import StandardScalerTorch, PCATorch
from urc.phi.image_phi import PhiImage

def numpy_pca(X, k):
    print("SDfsd")
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    C = Vt.T[:, :k]
    Z = Xc @ C
    return Z, X.mean(axis=0, keepdims=True), C

def test_scaler_matches_numpy():
    rng = np.random.default_rng(0)
    Xn = rng.normal(size=(512, 32)).astype(np.float32)
    Xt = t.tensor(Xn)

    sc = StandardScalerTorch().fit(Xt)
    Zt = sc.transform(Xt).cpu().numpy()

    # numpy baseline with population variance (ddof=0)
    mu = Xn.mean(axis=0, keepdims=True)
    var = ((Xn - mu)**2).mean(axis=0, keepdims=True)
    Zn = (Xn - mu) / np.sqrt(np.maximum(var, 1e-8))
    assert np.allclose(Zt, Zn, atol=1e-5), "Torch scaler should match NumPy"

def test_pca_matches_numpy():
    rng = np.random.default_rng(1)
    Xn = rng.normal(size=(400, 64)).astype(np.float32)
    Xt = t.tensor(Xn)

    pca_t = PCATorch(k=10).fit(Xt)
    Zt = pca_t.transform(Xt).cpu().numpy()

    Zn, mu, C = numpy_pca(Xn, 10)
    # Because of sign indeterminacy of singular vectors, compare subspace distances via reconstruction
    Xn_rec = Zn @ C.T + mu
    Xt_rec = pca_t.inverse_transform(pca_t.transform(Xt)).cpu().numpy()
    # Reconstruction error should be close
    err_t = np.mean((Xn - Xt_rec)**2)
    err_n = np.mean((Xn - Xn_rec)**2)
    assert abs(err_t - err_n) < 1e-4

def test_phi_end_to_end():
    device = "cuda" if t.cuda.is_available() else "cpu"
    X = t.randn(1024, 128, device=device)
    phi = PhiImage(backbone=None, d_out=128, pca_k=16).to(device)
    with t.no_grad():
        phi.fit_stats(X)
        Z = phi.project(X)
    assert Z.shape == (1024, 16)
    # basic stats finite
    assert t.isfinite(Z).all()
