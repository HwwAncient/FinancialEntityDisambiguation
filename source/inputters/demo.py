from source.inputters.corpus import *

"""
input 操作说明
"""

# 1. 需要数据首先需要实例化语料库corpus
# data_dir 参数需要指定data文件夹的相对位置
corpus = Corpus(data_dir='../../data')

# 2. 然后载入数据
corpus.load()

# 如果需要自己载入数据的调用
# corpus.build()

# build()读取数据存在 data/resource/demo.train demo.test
# 具体的格式为 text mention offset target 以'\t'作为分隔符，其中target为输出目标，如果匹配为1，不匹配为2

# 3. 获取DateLoader
# create_batches函数接受三个参数：
#               batch_size：batch的大小
#               data_type：loader的数据种类 只支持 `train` `test`
#               shuffle=False：是否随机获取数据
train_loader = corpus.create_batches(2, data_type='train')
test_loader = corpus.create_batches(2, data_type='test')

# 4. 获取数据
# DateLoader 是一个可以迭代的对象,迭代获取数据
# dateLoader 输出为一个四元组
#       input_text: 已经pad处理过的tensor[batch_size, max_length_sentence]
#       input_mention: tenser[batch_size]
#       input_offset: tenser[batch_size]
#       target: tenser[batch_size]
data_iter = iter(train_loader)
input_text, input_mention, input_offset, target = next(data_iter)
print(input_text, input_mention, input_offset, target)

# 5. 词嵌入
# corpus中的embeds属性是一个nn.Embedding对象, 在load时已经从文件加载了，可以直接使用
# 使用的词向量时预训练好的300维的词向量
embeds = corpus.embeds
embed_text = embeds(input_text)
print(embed_text)