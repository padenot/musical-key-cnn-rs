# Model provenance

`keynet.onnx` was exported from `checkpoints/keynet.pt` in
[a1ex90/MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN) at commit
`61d8fc170f262410f66a12057adffc5aceea0b2c`. The upstream project and checkpoint
are MIT-licensed.

`keynet.json` is the backend-independent RustNN graph lowered from that ONNX
model using `onnx2webnn` commit `d42b1adaf4120e78a5b22671c2fdc38c3f871bd3`.
`keynet.mlmodelc` is its ahead-of-time Core ML compilation. The scripts and exact
commands in the repository README reproduce both artifacts.

The preprocessing fixture was generated with `rosa` commit
`980f3509c6beb45b316f96cd59eca02e09ea025e`, which matches librosa's CQT
preprocessing for this model.
