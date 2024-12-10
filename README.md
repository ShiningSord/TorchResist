# LithoResistSimulator
A Neat Litho-Resist Simulator

The litho model part is forked from FuILT. We calibrate the model parameters on real designs.

We implemented the resist model from scratch and refined it with a calibration set.


# Installation 

一个step-by-step的教程，配置一个名为torchresist的AnaConda环境，注意使用markdown的代码环境，如
```
conda create -n torchresist python==3.9
conda deactivate
conda activate torchresist

pip3 install xxx
conda install xxxx
```
请参考 https://github.com/shelljane/lithobench 的对应部分的书写格式,并确保包的安装是对的(python我不确定是什么版本，你试试)

# Quick Start

先

``` 
git clone xxxx
```
## Prepare Mask


这里放个超链接导向 Lithobench的Github项目的某一行，让他们从这下mask数据，然后你要提供整理成我们需要的结构.

- 理想状态是在根目录下创建一个名为data的文件夹，存放各类数据。data文件夹下的各个子目录对应LithoBench中不同来源的Mask，如Dataset1,Dataset2. 注意，这里只需要未优化过的Mask就行了，不需要ILT的
- Dataset1下有一个mask文件夹，下有两个文件夹，Images和Numpys。 Images下按照mask000000.png的六位数字格式(从0开始)存放所有的二值化Binary图片。Numpys文件夹下，用numpy的array的`bool（重要）`格式，存放0-1的mask，shape应该是[B,H,W].
- 你需要提供一个脚本或者函数或者别的什么，等他们从LithoBench中下完数据以后，复制粘贴就能傻瓜式自动化的完成以上的事情，当然，只需要给一个例子就行，我们用那个16000+的那个例子。请把这个脚本存在根目录的tools下


提供一个脚本类似于 

```
bash tools/processmask.sh path/to/download.zip
```


最后应该长这样

```
data/dataset1/mask/1nm/imamges/mask000000.png
data/dataset1/mask/1nm/numpys/mask.npy
```

然后在readme中放几个mask的图示意，用于展示的图片存在本项目的demo/mask/下



### 7nm分辨率的mask (用文字标注，可选)


用文字标注为了提升效率，如果是用FuILT的litho模型，可以选择用7nm的mask,你提供一个脚本（存在tool下），对mask先做padding，然后降采样到7nm，存在

```
data/dataset1/mask/7nm/imamges/mask000000.png
data/dataset1/mask/7nm/numpys/mask.npy
```

脚本可以参考（/research/d5/gds/zxwang22/code/LithoResistSimulator/tools/downsampling.py）

## Litho Simulation

这里我们提供两种 Litho Model的方案，ICCAD13和 FuILT

文件格式应为，根目录/lithosimulators/ICCAD13 和根目录/lithosimulators/FuILT，把对应的simulator存到对应的目录下







### 对于ICCAD13

1. 请cite ICCAD13的那个光学模型的paper和Lithobench，用文字说明我们这里的litho model的code来源于lithobench的对应代码，并且根据之前的paper(LithoBench)，这里使用的mask的分辨率固定为1nm
2. 准备一个脚本同样存在tools下，把上面准备好的1nm Mask，自动地进行padding，litho simulation，然后降采样到7nm的分辨率，最后把这个降采样的结果存到data中，存储的路径应为
```
data/dataset1/litho/iccad13/images/litho000000.png
data/dataset1/litho/iccad13/numpys/litho.npy
```

这里我发现可以用plt.imsave()方便地存储numpy的图像化结果，推荐你用这个


提供一个代码类似于



```
python3 litho_iccad13.py --mask path/to/1nm/mask/numpy.npy --outpath xxxx
```



### 对于FuILT

类似的事情再做一次,结果存在 （这里的code可以参考/research/d5/gds/zxwang22/code/LithoResistSimulator/eval/abbe_batch.py）

```
data/dataset1/litho/fuilt/images/litho000000.png
data/dataset1/litho/fuilt/numpys/litho.npy
```

`额外用文字说明： FuILT的输入mask分辨率可选7nm，通过一个参数控制`



提供一个代码类似于

```
python3 litho_fuilt.py --mask path/to/1nm/mask/numpy.npy --resolution 1.0 --outpath xxxx
python3 litho_fuilt.py --mask path/to/7nm/mask/numpy.npy --resolution 7.0 --outpath xxxx
```


### 可能的第三种方案

请暂时留空，我还在申请权限，如果能行，我们应该放一个png和一个numpy


对于mask中展示的demo，分别展示其对应的光学结果，也放在本项目的demo/litho/ICCAD13/,demo/litho/FuILT/下




## TorchResist

TorchResist is a powerful tool designed to provide resist parameters for a wide range of optical lithography solutions. For each optical setup, we offer a comprehensive set of resist parameters and an easy-to-use script to perform resist simulations.

### Usage
You can use the provided script to simulate resist by specifying the necessary parameters. For example:

```
python3 -m examples.resist --lithomodel FUILT --lithoresult /research/d5/gds/zxwang22/storage/resist/fuilt/lithoresult/7nm/abbe_intensity.npy --outpath /research/d5/gds/zxwang22/storage/resist/fuilt/resist --resolution 1.0
```

- `--lithomodel`: Specifies the used lithography model (chosen from `ICCAD13` and `FUILT`).
- `--lithoresult`: Path to the input `.npy` file containing lithography results.
- `--outpath`: Directory to save the output files.
- `--resolution`: Resolution of input in nanometers (default: `1.0`).


### Features

- **Customizable Parameters:** Adjust resist settings via input arguments for different lithography models and resolutions.
- **Flexible Resolution:** By default, the tool assumes a resolution of 7nm. While this resolution has not been rigorously validated, you can modify it based on your requirements without significantly impacting the results.




类似的，加入demo展示


# TO: Jieya 

最后要试试，按照你的傻瓜式操作，能从0开始得到想要的结果

有任何问题，直接问我



# Extention 高级的使用技巧，我来