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


预训练模型存放于pretrained目录（使用aclimdb数据集预训练）：   <br /> 
提取码：   <br />  <br /> 


训练：
CUDA_VISIBLE_DEVICES=0 python train.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B  <br /> 
评估：
CUDA_VISIBLE_DEVICES=0 python eval.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B --ckpt_path outputs/dpcnn-20_149.ckpt
 <br />  <br /> 

使用预训练模型进行调优，训练结果精度：



