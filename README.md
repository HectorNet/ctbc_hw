# Handwriting recognition for Traditional Chinese name

## Data
資料集出處：https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

整理過的資料集可下載於: https://drive.google.com/file/d/1Hu7iy8fr5rqaFq73ukzQBaBL-BJymGMC/view?usp=sharing

此資料集包含4803個字元，利用這些字元做為手寫姓名的組合，其中姓氏字元僅取台灣的百大姓氏(https://taiwan.chtsai.org/2006/01/10/taiwan_baijiaxing/)。

由於作答時間有限的關係，本project將做下列的限制：
1. 僅限姓名長度為3的情況
2. 僅處理橫寫姓名(由左至右)
3. 模型輸入須為150x50像素

example:
![白佳奇](https://github.com/HectorNet/ctbc_hw/blob/dev/data/%E7%99%BD%E4%BD%B3%E5%A5%87.png)


## Model (利用PyTorch實作)
Input shape: [-1, 1, 50, 150]

Output shape: [-1, 4803, 1, 3]

此模型使用Resnet block做了五層的堆疊，其中包含了兩次stride=2的convolution做為downsample及最後利用adaptive average pooling得到output。

後處理中，將模型的output經argmax(dim=1)後，可得到最大機率的三個字元。

Model summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 25, 75]             320
       BatchNorm2d-2           [-1, 32, 25, 75]              64
              ReLU-3           [-1, 32, 25, 75]               0
            Conv2d-4           [-1, 32, 25, 75]           9,248
       BatchNorm2d-5           [-1, 32, 25, 75]              64
            Conv2d-6           [-1, 32, 25, 75]             320
              ReLU-7           [-1, 32, 25, 75]               0
          ResBlock-8           [-1, 32, 25, 75]               0
            Conv2d-9          [-1, 128, 25, 75]          36,992
      BatchNorm2d-10          [-1, 128, 25, 75]             256
             ReLU-11          [-1, 128, 25, 75]               0
           Conv2d-12          [-1, 128, 25, 75]         147,584
      BatchNorm2d-13          [-1, 128, 25, 75]             256
           Conv2d-14          [-1, 128, 25, 75]          36,992
             ReLU-15          [-1, 128, 25, 75]               0
         ResBlock-16          [-1, 128, 25, 75]               0
           Conv2d-17          [-1, 512, 25, 75]         590,336
      BatchNorm2d-18          [-1, 512, 25, 75]           1,024
             ReLU-19          [-1, 512, 25, 75]               0
           Conv2d-20          [-1, 512, 25, 75]       2,359,808
      BatchNorm2d-21          [-1, 512, 25, 75]           1,024
           Conv2d-22          [-1, 512, 25, 75]         590,336
             ReLU-23          [-1, 512, 25, 75]               0
         ResBlock-24          [-1, 512, 25, 75]               0
           Conv2d-25         [-1, 1024, 13, 38]       4,719,616
      BatchNorm2d-26         [-1, 1024, 13, 38]           2,048
             ReLU-27         [-1, 1024, 13, 38]               0
           Conv2d-28         [-1, 1024, 13, 38]       9,438,208
      BatchNorm2d-29         [-1, 1024, 13, 38]           2,048
           Conv2d-30         [-1, 1024, 13, 38]       4,719,616
             ReLU-31         [-1, 1024, 13, 38]               0
         ResBlock-32         [-1, 1024, 13, 38]               0
           Conv2d-33         [-1, 2048, 13, 38]      18,876,416
      BatchNorm2d-34         [-1, 2048, 13, 38]           4,096
             ReLU-35         [-1, 2048, 13, 38]               0
           Conv2d-36         [-1, 2048, 13, 38]      37,750,784
      BatchNorm2d-37         [-1, 2048, 13, 38]           4,096
           Conv2d-38         [-1, 2048, 13, 38]      18,876,416
             ReLU-39         [-1, 2048, 13, 38]               0
         ResBlock-40         [-1, 2048, 13, 38]               0
           Conv2d-41         [-1, 4803, 13, 38]      88,533,699
             ReLU-42         [-1, 4803, 13, 38]               0
AdaptiveAvgPool2d-43           [-1, 4803, 1, 3]               0
          Softmax-44           [-1, 4803, 1, 3]               0
================================================================
Total params: 186,701,667
Trainable params: 186,701,667
Non-trainable params: 0
----------------------------------------------------------------
```

## Training
*由於訓練時間有限，在此僅使用10個字元做為子資料集(data/sample-train與data/sample-test)，而上述ouput shape應修正為[-1, 10, 1, 3]。*

`python train.py --num_train_examples 10000 --num_test_example 100 --batch_size 32 --epochs 200 --log_freq 100 --save_freq 1`

## Test
在第三個epoch時，模型訓練在測試資料集(data/sample-test)的準確度已達到將近100%。

`python inference.py`
