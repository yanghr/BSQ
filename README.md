# CIFAR-10 experiments

This folder contains the code for inducing mixed precision quantization schemes with BSQ on the CIFAR-10 dataset. Multiple common model architectures and layer configurations used on CIFAR-10 are available. However only ResNet models are configured to support BSQ training with bit representation, so as to achieve the results in the main paper.

## Acknowledgement

The training and evaluation codes and the model architectures are adapted from [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).

## Specification of dependencies

This code is tested with Python 3.6.8, PyTorch 1.2.0 and TorchVision 0.4.0. It is recommanded to use the provided `spec-file.txt` file to replicate the anaconda environment used for testing this code, which can be done by:

```
conda create --name myenv --file spec-file.txt
```

We suggest using GPU to run this code for the best efficiency. Both running on a single GPU or running in parallel on multiple GPUs are supported.

## Usage

### Pretrained models 

As introduced in Appendix A.1, pretrained models are used to initiate the BSQ training. The pretrained model are provided in the `\checkpoints\cifar10\` folder, where the checkpoint in `resnet-20\` is the full-precision pretrained model and the checkpoint in `resnet-20-8\` is the 8-bit quantized model in bit representation. 

For more details on training the full-precision model please see the training recipes provided by [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md ). The quantized model is achieved with `convert.py`, which will be introduced later.

### BSQ training

Here we perform BSQ training on the ResNet-20 model on the CIFAR-10 dataset.

```
python cifar_prune_STE.py -a resnet --depth 20 --epochs 350 --lr 0.1 --schedule 250 --gamma 0.1 --wd 1e-4 --model checkpoints/cifar10/resnet-20-8/model_best.pth.tar --decay 0.01 --Prun_Int 50 --thre 0.0 --checkpoint checkpoints/cifar10/xxx --Nbits 8 --bin --L1 >xxx.txt
```

`xxx` in the command should be replaced with the folder you want for saving the achieved model. The achieved model will be saved in bit representation. We suggest redirecting the print output to a txt file with `>xxx.txt` to avoid messing up with the progress bar display and keep record of the training process. 

`--decay` is used to set the regularization strength $$\alpha$$ in Equation (5), so as to explore the accuracy-model size tradeoff. Results for using different $$\alpha$$ are shown in Section 4.2.

`--Prun_Int` is the number of epochs between each re-quantization and precision adjustment step, which is suggested to be set to 50. The effect of using a smaller interval is illustrated in Appendix B.1.


### Evaluating and finetuning achieved model

The model achieved from BSQ training can be evaluated and finetuned with `cifar_finetune.py`.

For evaluation, run

```
python cifar_finetune.py -a resnet --depth 20 --model checkpoints/cifar10/xxx/checkpoint.pth.tar --Nbits 8 --bin --evaluate
```
`xxx` in the command should be replaced with the folder used to save the BSQ trained model. Note that only model in bit representation can be evaluated in this way. The testing accuracy, the precentage of 1s in each bit of each layer's weight and the precision assigned to each layer will be printed in the output. 

To further finetune the ahcieved model, use

```
python cifar_finetune.py -a resnet --depth 20 --epochs 250 --lr 0.0001 --schedule 250 --gamma 0.1 --wd 1e-4 --model checkpoints/cifar10/xxx/checkpoint.pth.tar --checkpoint checkpoints/cifar10/xxx-ft --Nbits 8 --bin >xxx-ft.txt
```

The quantization scheme will be fixed throughout the finetuning process. At the end of finetuning, the model with the highest testing accuracy will be stored in both bit representation and floating-point weights. The bit representation is saved in `checkpoints/cifar10/xxx-ft/best_bin.pth.tar` and the floating-point model is saved in `checkpoints/cifar10/xxx-ft/best_float.pth.tar`


### Converting full-precision models to bit representation with achieved quantization schemes

To convert a full-precision model, use

```
python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar --dict checkpoints/cifar10/xxx/checkpoint.pth.tar --checkpoint checkpoints/cifar10/xxx-mp --Nbits 8 >xxx-mp.txt
```
 
If the path in `--dict` is provided, the model will be converted to the same quantization scheme as the model specified in `--dict`. Otherwise the whole model will be quantized to the precision specified in `--Nbits`. The converted model will be in bit representation, and will be saved in the folder specified in `--checkpoint`. We use this code to achieve the 8-bit quantized model before BSQ training, and to achieve the "train from scratch" models that are further finetuned to be compared in Table 1.
 
 

