# 古诗词生成GPT


implementation of a naive transformer and a modern transformer model with 
1. RoPE
2. RMSNorm
3. Group Query Attention
4. GLU
5. K, V cache

# Implementation Details

## 训练数据
https://github.com/THUNLP-AIPoet/Datasets/tree/master/CCPC
Training 109,727; Validation 7,979; Testing: 9,976

use `prepare_data.py` to build vocab and pretokenize data.

you can download pretokenized data in [here](https://www.kaggle.com/datasets/taoshao/chinese-poetry)

## vocab
使用了character based分词方式，每个中文字符为一个token，词表的大小为5000，加上4个特殊token：\<SOS\>, \<EOS\>, \<UNK\>, \<PAD\>，一共5004，未出现的字符换做\<UNK\>。

## features
* RoPE： 旋转位置编码，将每个token embedding两两分为一组，不同的组以不同频率的旋转矩阵进行操作，长度外推性能很好。
* RMSNorm：只保留layernorm中的scale操作，去除了center的操作，减小计算量。
* GLU：具有f(x)*g(x)形式的激活函数。
* GQA：减小了k，v cache的大小，减小推理的显存。
* kv cache：减小推理计算量。

## training
* use `run.sh` to lanuch run.
* adjust config in `configs/xxx_config.py`
* 由于数据量较小，需要较大的weight decay(1.0)来防止过拟合，否则会出现验证集损失先降低后升高的情况。




# Results
只提供开头的第一个字，例如春，夏，秋，冬

* 春风门内多新事|花白年年亦老我|岂怪春深分过云|只知造化须相并
* 夏盘且登并|秋崖莫惊看|多为湍响走|行有省笑难
* 秋留云月主人家|久住长筇更自重|过雨满朝浑不见|水光一带向何通
* 冬月将那不足看|风前一夜动朝寒|栽花莫为知明道|老蕊都须露里看
* [去年今日]上河山|城谷风尘等等闲|今日片帆犹喜至|悔持灯话涉江湾

# TODO
* implement modern tokenizer: e.g BPE

