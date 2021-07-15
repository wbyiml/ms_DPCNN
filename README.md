# ms_task1

## wbyiml/ms_task1 


数据集存放在resource目录内 <br />
使用resource/makeDataset.py将数据集转换为train.txt, test.txt <br /> 
resourece <br /> 
——glove.6B <br /> 
————glove.6B.{50-300}d.txt   <br /> 
——rt-polaritydata   <br /> 
————processed   （训练时产生的预处理文件） <br /> 
————rt-polarity.pos   数据集正样本 <br /> 
————rt-polarity.neg   数据集负样本 <br /> 
————train.txt     makeDataset.py生成 <br /> 
————test.txt      makeDataset.py生成 <br />  <br /> 


glove embedding文件：https://pan.baidu.com/s/14BvOavIY0IqMfbJlU0_rZw <br /> 
提取码：nyxg   <br />  <br /> 

预训练模型：https://pan.baidu.com/s/1LGL1_N75HDO3VHmFchY7UA <br /> 
提取码：zrs1   <br />  <br />

训练：
CUDA_VISIBLE_DEVICES=0 python train.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B  <br /> 
使用预训练模型训练：
CUDA_VISIBLE_DEVICES=0 python train.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B --pretrained pretrained/aclimdb.ckpt  <br /> 
评估：
CUDA_VISIBLE_DEVICES=0 python eval.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B --ckpt_path outputs/dpcnn-20_149.ckpt
 <br />  <br /> 

使用预训练模型进行调优，训练结果精度：0.779  <br /> 
结果模型：https://pan.baidu.com/s/1bd3vi6Z4kq4LzmA7LQL1LA <br /> 
提取码：mic9   <br />  <br />



