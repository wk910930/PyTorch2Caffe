# PyTorch2Caffe
A converter between PyTorch and Caffe.

## TODO

- [ ] Parse PyTroch model
- [ ] Auto-generate/Manually-specify prototxt files used by Caffe
- [ ] Convert a toy model (i.e. with only basic layers such as `conv`, `relu` and `fc`)
- [ ] Add more options, such as whether using `bias` term
- [ ] Convert AlexNet, VGGNet
- [ ] Support `BatchNorm Layer`
- [ ] Support different pre-processing (e.g. PyTorch will normalize/scale the intput image to `[0, 1]`, while Caffe usually doesn't)
- [ ] Convert ResNet-18/50/101
- [ ] All Done
