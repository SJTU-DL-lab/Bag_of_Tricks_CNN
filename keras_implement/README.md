# tricks-in-deeplearning
Using different tricks to improve performance of resetnet by Keras

Paper：https://arxiv.org/abs/1812.01187

Resnet model and other CNN models implemented by keras can be found from: https://github.com/BIGBALLON/cifar-10-cnn

*basenet_resnet.py: baseline*

*cosine_batch.py: adapt lr by batch you can custome your own lr stradegy according to the demo*

*mix_generator.py: mix up*

*resnet110.py: the final training code*

Train baseline Resnet32 model （done)  accuracy: 91.64%

Adding warmup LR (done) accracy:92.32%(+0.68%)

Adding cosine decay（done) accuracy:93.01%（+0.69%）

Adding cosine decay based on batch (done). But it does not improve for accuracy:92.93%

Adding mixup(done) accuracy:94.10%(+1.09%)

I tried label smoothing but it does not improve. According to https://www.researchgate.net/publication/327004087_Empirical_study_on_label_smoothing_in_neural_networks, label smoothing is not suitable for cifar 10.

Using smaller batch size :accuracy 94.38%(+0.28%)

Using resnet110: accuracy 95.21%(+0.83%)
