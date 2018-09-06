"""
TACOTRON: A FULLY END-TO-END TEXT-TO-SPEECH SYNTHESIS MODEL
摘要：
  文本到语音的合成系统通常包含好几个阶段，比如前端的文本分析，声学模型和语音合成模型。这些通常需要很广泛且专业的知识。本文提出Tacotron，一个端到端的从文本生成语音的模型。给定<text，audio>对，可以通过随机初始化从头开始完全训练模型。此外，由于Tacotron在帧级别生成语音，因此它比样本级别的自回归方法快得多。
介绍：
文本到语音的过程很复杂。例如， TTS通常具有提取各种语言特征的文本前端，持续时间模型，声学特征预测模型和基于复杂信号处理的声码器。这些组件基于广泛的领域专业知识，并且设计费力。 它们也是独立训练的，因此每个组件的错误可能会复合。 因此，现代TTS设计的复杂性导致在构建新系统时的大量工程努力。
因此，集成的端到端TTS系统具有许多优点，可以在具有最少人类注释的<text，audio>对上进行训练。 首先，这样的系统减少了繁重的特征工程的需要。其次，它更容易对各种属性如说话者或语言或情绪等高级功能进行丰富的调节。 这是因为调节可以在模型的最开始发生，而不是仅发生在某些组件上。 同样，适应新数据也可能更容易。 最后，单个模型可能比多阶段模型更稳健，其中每个组件的错误可以复合。 这些优势意味着端到端模型可以让我们在现实世界中丰富的，富有表现力但经常是嘈杂的数据上进行大规模的训练。
TTS是一个大规模的反问题：高度压缩的源（文本）被“解压缩”为音频。 由于相同的文本可以对应于不同的发音或说话风格，这对于端到端模型来说是特别困难的学习任务。 此外，与端到端语音识别或机器翻译不同，TTS输出是连续的，并且输出序列通常比输入的序列长得多。 这些属性导致预测错误快速累积。
在本文中，我们提出了Tacotron，一种基于序列到序列（seq2seq）的端到端生成TTS的模型，具有注意力范例。 我们的模型将字符作为输入并输出原始频谱图，使用几种技术来提高vanilla seq2seq模型的能力。给定<text，audio>对，Tacotron可以通过随机初始化从头开始完全训练。 它不需要音素级对齐，因此可以轻松扩展为使用大量的声学数据和文本。 通过简单的波形合成技术，Tacotron在美国英语评估集上产生3.82平均意见得分（MOS），在自然性方面优于production parametric system。
相关工作：
WaveNet是一种强大的音频生成模型。 它适用于TTS，但由于其样本级别的自回归性质而很慢。它还需要对现有TTS前端的语言特征进行调节，因此不是端到端：它只取代声码器和声学模型。 另一个最近开发的神经模型是DeepVoice，它通过相应的神经网络替换典型TTS pipeline中的每个组件。 然而，每个组件都经过独立培训，将系统更改为以端到端的方式进行培训并不重要。
据我们所知，Wang et al 是最早使用seq2seq with attention接触端到端TTS的工作。 但是，它需要预先训练的隐马尔可夫模型（HMM）对准器来帮助seq2seq模型学习对齐。 很难说seq2seq本身学到了多少对齐。其次，使用一些技巧来训练模型，作者注意到它会伤害韵律。 第三，它预测声码器参数因此需要声码器。 此外，该模型在音素输入上进行训练，实验结果似乎有些限制。
Char2Wav是一个独立开发的端到端模型，可以在字符上训练。 然而，在使用SampleRNN神经声码器之前，Char2Wav仍然预测声码器参数，而Tacotron直接预测原始频谱图。 此外，他们的seq2seq和SampleRNN模型需要单独预先训练，但我们的模型可以从头开始训练。 最后，我们对vanilla seq2seq范例进行了几项关键修改。 如后所示，vanilla seq2seq模型不适用于字符级输入。
模型结构：
    Tacotron是一个基于注意力的seq2seq模型。 图1描绘了该模型，其包括编码器，基于注意力的解码器和后处理网络。我们的模型将字符作为输入并生成频谱图帧，然后将其转换为波形。
 
 
3.1 CBHG模块
CBHG由一组1-D卷积滤波器组成，其次是highway network和双向门控递归单元（GRU）递归神经网络（RNN）。CBHG是一个功能强大的模块，用于从序列中提取特征表达。首先用K组1-D卷积滤波器对输入序列进行卷积，其中第k组包含宽度为k的Ck滤波器（即k = 1; 2; :::; K）。 这些滤波器明确地模拟了本地和上下文信息（类似于建模unigrams，bigrams，到K-gram）。卷积输出堆叠在一起，并且沿着时间方向进行max pool，以增加局部不变性。 请注意，我们使用步长1来保留原始时间分辨率。 我们进一步将处理后的序列传递给几个固定宽度的1-D卷积，其输出通过residual connection与原始输入序列相加。Batch Normalization用于所有卷积层。 卷积输出被馈送到多层的high-way network以提取高级特征。最后，我们在顶部堆叠双向GRU RNN以从前向和后向上下文中提取顺序特征。 CBHG受到机器翻译工作的启发，其中与Lee等人的主要区别在于使用non-causal convolutions, batch normalization, residual connections, and stride=1 max pooling. 我们发现这些修改改进了网络的泛化能力。
3.2 编码器
    编码器的目标是提取文本的强健顺序表示。编码器的输入是字符序列，其中每个字符被表示为one-hot向量并且被嵌入到连续矢量中。 然后，我们对每个embedding进行一系列非线性变换，统称为“pre-net”。 我们把带有dropout的bottleneck层作为pre-net，这有助于收敛并改善泛化。 CBHG模块将pre-net的输出转换为用于注意力模块的编码器表达。我们发现这种基于CBHG的编码器不仅减少了过度拟合，而且比标准的多层RNN编码器产生更少的错误发音（参见我们的音频样本链接页面）。
3.3 解码器
我们使用基于内容的tanh注意解码器，其中状态重复层在每个解码器时间步骤产生注意力查询。我们连接上下文矢量和注意力RNN信元输出以形成解码器RNN的输入。我们将一堆具有垂直residual connection的GRU用于解码器。我们发现residual connection加速了收敛。
解码器目标是一个重要的设计选择。虽然我们可以直接预测原始频谱图，但是用于学习语音信号和文本之间的对齐（这实际上是使用seq2seq完成此任务的动机），这是一种高度冗余的表示。由于这种冗余，我们使用不同的目标进行seq2seq解码和波形合成。 seq2seq目标可以高度压缩，只要它为反演过程提供足够的可懂度和韵律信息。我们使用80波段的mel-scale声谱作为目标，但可以使用更少的波段或更简洁的目标，如倒谱。我们使用post-processing network 将seq2seq target转换为波形。
我们使用简单的全连接输出层来预测解码器目标。我们发现的一个重要技巧是在每个解码器步骤预测多个非重叠输出帧。一次预测 r 帧将解码器步骤的总数除以r，这减小了模型大小，训练时间和推理时间。 更重要的是，我们发现这一技巧可以大幅提高收敛速度，这可以通过从注意力中学到更快（更稳定）的对齐来衡量。这可能是因为相邻语音帧是相关的，并且每个字符通常对应于多个帧。 一次发射一帧会强制模型在多个时间步长内处理相同的输入令牌; 发射多个帧允许注意力在训练早期向前推进。 Zen等人也使用了类似的技巧，但主要是为了加快推理。
解码器的第一步骤以全零帧为条件，其表示<GO>帧。 在推断中，在解码器步骤t，最后一帧的r个预测将被输入到解码器的t+1步。 请注意，提供最后一次预测是一个临时选择 - 我们可以使用所有r预测。 在训练期间，我们总是将每个第r个ground truth frame送到解码器。 输入帧如编码器中那样传递到pre-net。 由于我们不使用预定采样等技术（我们发现它会损害音频质量），pre-net中的dropout对模型的推广至关重要，因为它提供了一个噪声源来解决输出分布中的多种模态。
3.4 POST-PROCESSING NET 和波形合成
    post-processing net的任务是将seq2seq target转换为可以合成为波形的目标。由于我们是用Griffin-Lim作为合成器，post-processing net学习预测在线性频率范围内采样的频谱幅度。post-processing net的另一个动机是它可以看到完整的解码序列。与始终从左到右运行的seq2seq相比，它具有前向和后向信息以校正每个单独帧的预测误差。在这项工作中，我们使用CBHG模块进行后处理网络，尽管更简单的架构可能也可以。post-processing net的概念非常普遍。 它可用于预测替代声码器参数，或作为类似WaveNet的神经声码器直接合成波形样本。
我们使用Griffin-Lim算法从预测的频谱中合成波形。 我们发现，在喂入Griffin-Lim之前，将预测的幅度提高1.2，可以减少伪影，这可能是由于其谐波增强效应。 我们观察到Griffin-Lim在50次迭代后收敛（事实上，大约30次迭代似乎就足够了），这相当快。 我们在TensorFlow中完成了Griffin-Lim，因此它也是该模型的一部分。 虽然Griffin-Lim是可以区分的（它没有可训练的权重），但我们并没有在这项工作中给它带来任何损失。我们选择Griffin-Lim是为了简单起见，虽然它已经产生了强大的结果，但一种快速和高质量的可训练频谱到波形逆变器正在开发中。
4 模型的细节
我们使用对数幅度谱图与Hann窗口，50ms帧长度，12.5ms帧移位和2048点傅立叶变换。 我们还发现预强调（0.97）是有帮助的。 我们对所有实验使用24kHz采样率。
我们在本文中使用r = 2（输出层缩减因子）作为MOS结果，尽管较大的r值（例如r = 5）也能很好地工作。 我们使用Adam优化器，学习率衰减，从0.001开始，分别在500K，1M和2M步骤后减少到0.0005,0.0003和0.0001。 我们对seq2seq解码器（mel频谱）和post-processing net（线性频谱）使用简单的l1损失。这两个损失的权重相等。
我们每个batch使用32个样本进行训练，其中所有序列都填充到最大长度。 通常使用loss mask训练序列模型，这掩盖了零填充帧上的损失。 然而，我们发现以这种方式训练的模型不知道何时停止输出，导致重复声音到达终点。 解决这个问题的一个简单技巧是重建零填充帧。
 
"""