如何复现代码？（以RD001-->RD003为例子）

首先我们需要将args.data_fn设置为1，args.datatest_fn设置为1，进行模型的训练。
此时在model文件夹中保存的模型均是元学习训练得到的模型，若目录文件夹中名中support size=0,说明该模型是正常训练得到的。
注意这个过程如果重新生成数据的话，生成过程比较缓慢。

(测试阶段)
保存模型后，我们要将args.do_train设置为False,避免重新训练模型，从上述得到的文件夹中load模型，因此需要注意load的模型对应的文件夹名称。

将args.data_fn设置为3，args.datatest_fn设置为3. 此时的train dataset是RD003的，这个会作为referrence出现，测试集也是RD003.
注意此时的模型是load在RD001训练的模型，测试在RD003上。此时便实现了我们的目的，利用在FD001上训练的元模型（support size≠0）和正常模型（support size=0），迁移到FD003.

迁移时需要注意：
1. Base Model直接迁移。此时如何设置，load support_size=0的正常训练的模型，在测试时也需要设置support size=0（表示不微调）
2. base Model+Finetune。此时的设置仍然是load support_size=0的正常训练的模型，在测试时设置support size≠0，support size的大小就是选择多少个样本进行微调
3. 元学习模型+Finetune。此时load support size≠0的元学习模型，在测试时设置support size≠0。
4. 
需要修改什么位置？
在测试阶段时，需要修改两个位置，一个是load模型的位置，一个是输出结果的位置（需要密切注意）

#强调
该代码整理时间有点晚，因此有些细节记忆的不要清楚，需要重新阅读。

run脚本（可能不准确）：
目前run1_retrain.sh为训练最新版本（do_train）也是元学习测试，和baseModel直接测试的最新版本（do_eval） 
run1_eval.sh是Basemodel+finetune的测试，注意main.py load模型需要注意









