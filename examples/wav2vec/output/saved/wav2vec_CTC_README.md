第一版的wav2vec模型使用了400+小时的无监督音频来训练，以英文版的wav2vec_vox.pt为微调的起点，从checkpoint63.pt开始微调，到checkpoint83.pt结束，最好的节点是checkpoint80.pt。
ctc1: 训练数据为220小时的数据堂对话音频，使用第一版wav2vec模型来进行语音表示，训练中loss一直震荡，训练失败，训练中使用了"w2l_args"关键字。
ctc2: 训练数据为220小时的数据堂对话音频，使用第一版wav2vec模型来进行语音表示，训练收敛，训练了37个epochs,最好的节点是checkpoint36.pt.训练中未使用"w2l_args"关键字。最优模型在yitu_chat的CER=29.19, yitu-reverb的CER=41.8525. best.pt 来解码， yitu-reverb的CER=100. 从训练日志中的loss判断，大约12个epochs,模型已基本收敛。
ctc3: 训练数据为200+小时的音频，称为all20, 是等比例的从原始全部监督训练数据集上抽取的，抽取了20%。使用第一版wav2vec模型来进行语音表示，训练未收敛
ctc4: 训练数据all20, 是等比例的从原始全部监督训练数据集上抽取的，抽取了20%。使用第一版wav2vec模型来进行语音表示，训练收敛，共训练了17个epochs,最优的节点是checkpoint17.pt. 使用checkpoint16.pt测试yitu-chat,cer=17.675.相比于数据堂的29.19，有显著提升。
ctc5: 训练数据all20, 注意它的wav2vec模型使用的是原始的英文的wav2vec_vox.pt，训练了20个epochs, 最优的节点是checkpoint20.pt, 使用该最优节点来测试yitu-chat CER=18.9149，可见与ctc4里的最优表现差距不大。
ctc6:训练数据all50, 是等比例的从原始全部监督训练数据集上抽取的，抽取了50%。使用第一版wav2vec模型来进行语音表示，训练从ctc4里的最优节点开始(checkpoint18.pt), 训练到checkpoint24.pt, 训练了6个epochs, 最优节点是checkpoint24.pt, 测试 yitu-chat数据集，CER=16.6837. 测试yitu-reverb, CER: 28.8147。
ctc7:训练数据all1, 是等比例的从原始全部监督训练数据集上抽取的，抽取了1%, 大约10小时。wav2vec模型使用的是英文原版的wav2vec_small.pt
