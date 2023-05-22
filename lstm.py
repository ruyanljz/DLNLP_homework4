import re
import jieba
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from tensorflow import keras


def data_processing(file_path):
    """
    获取文件信息并预处理
    :param file_path: 文件名对应路径
    :return data_out: 字符串形式的语料库
    :return words_out: 分词
    """
    # 读取小说
    delete_symbol = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@★、…【】《》‘’[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='ANSI') as f:
        data_out = f.read()
        data_out = data_out.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data_out = re.sub(delete_symbol, '', data_out)
        data_out = data_out.replace('\n', '')
        data_out = data_out.replace('\u3000', '')
        data_out = data_out.replace(' ', '')
        f.close()
    # 以词为单位进行分词
    words_out = list(jieba.cut(data_out))
    return data_out, words_out


def lstm(words_in):
    """
    获取文件信息并预处理
    :param words_in: 数据集
    """
    # 创建字符与数字的映射
    chars = list(set(words_in))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    # 统计数据集中的字符数和词汇量
    n_chars = len(words_in)
    n_vocab = len(chars)

    # 设定LSTM模型的参数
    seq_length = 60
    step = 3
    X_data = []
    y_data = []
    for i in range(0, n_chars - seq_length, step):
        seq_in = words_in[i:i + seq_length]
        seq_out = words_in[i + seq_length]
        X_data.append([char_to_int[char] for char in seq_in])
        y_data.append(char_to_int[seq_out])
    n_patterns = len(X_data)

    # 将输入数据转换为LSTM模型需要的格式
    X = np.reshape(X_data, (n_patterns, seq_length, 1))
    y = []
    for i in y_data:
        temp = [0] * n_vocab
        temp[i] = 1
        y.append(temp)
    y = np.array(y)

    # 创建LSTM模型
    model = Sequential()
    model.add(Embedding(y.shape[1], 256))
    # model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # 定义模型的检查点
    # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
            filepath='text_gen.h5',
            monitor='loss',
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(  # 不再改善时降低学习率
            monitor='loss',
            factor=0.5,
            patience=1,
        ),
        keras.callbacks.EarlyStopping(  # 不再改善时中断训练
            monitor='loss',
            patience=3,
        ),
    ]

    # 训练LSTM模型
    model.fit(X, y, epochs=200, callbacks=callbacks_list)

    # 使用训练好的模型生成新的文本
    num = 200
    text = '这少女这两下轻轻巧巧的刺出，戳腕伤目，行若无事，不知如何，那吴国剑士竟是避让不过。余下七名吴士大吃一惊，一名身材魁梧的吴士提起长剑，剑尖也往少女左眼刺去。剑招嗤嗤有声，足见这一剑劲力十足。'
    # text = '两名剑士各自倒转剑尖，右手握剑柄，左手搭于右手手背，躬身行礼。两人身子尚未站直，突然间白光闪动，跟着铮的一声响，双剑相交，两人各退一步。旁观众人都是“咦”的一声轻呼。'
    text_cut = list(jieba.cut(text))[:seq_length]
    pattern = [char_to_int[char] for char in text_cut]
    print("Seed:")
    print(''.join(text_cut))
    for i in range(num):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = chars[index]
        print(result, end='')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]


if __name__ == "__main__":
    files = ['./data_novel/白马啸西风.txt',
             './data_novel/碧血剑.txt',
             './data_novel/飞狐外传.txt',
             './data_novel/连城诀.txt',
             './data_novel/鹿鼎记.txt',
             './data_novel/三十三剑客图.txt',
             './data_novel/射雕英雄传.txt',
             './data_novel/神雕侠侣.txt',
             './data_novel/书剑恩仇录.txt',
             './data_novel/天龙八部.txt',
             './data_novel/侠客行.txt',
             './data_novel/笑傲江湖.txt',
             './data_novel/雪山飞狐.txt',
             './data_novel/倚天屠龙记.txt',
             './data_novel/鸳鸯刀.txt',
             './data_novel/越女剑.txt']
    files_inf = ["白马啸西风", "碧血剑", "飞狐外传", "连城诀", "鹿鼎记", "三十三剑客图", "射雕英雄传", "神雕侠侣",
                 "书剑恩仇录", "天龙八部", "侠客行", "笑傲江湖", "雪山飞狐", "倚天屠龙记", "鸳鸯刀", "越女剑"]

    choice = 15  # 选择要训练的小说 范围0-15
    print('\n训练集为小说：' + '《' + files_inf[choice] + '》')
    data, words = data_processing(files[choice])  # 数据处理
    lstm(words)
