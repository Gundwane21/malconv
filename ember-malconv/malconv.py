#!/usr/bin/python
'''defines the MalConv architecture.
Adapted from https://arxiv.org/pdf/1710.09435.pdf
Things different about our implementation and that of the original paper:
 * The paper uses batch_size = 256 and SGD(lr=0.01, momentum=0.9, decay=UNDISCLOSED, nesterov=True )
 * The paper didn't have a special EOF symbol
 * The paper allowed for up to 2MB malware sizes, we use 1.0MB because of memory on a Titan X
 '''
num_of_classes = 9
def main(): 
    from keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply
    from keras.models import Model
    from keras import backend as K
    from keras import metrics
    from tensorflow.keras.utils import to_categorical
    import multi_gpu
    import os
    import math
    import random
    import argparse
    import os
    import numpy as np
    import requests

    batch_size = 50
    input_dim = 257 # every byte plus a special padding symbol
    padding_char = 256
    

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='number of GPUs', default=1)

    args = parser.parse_args()
    ngpus = int(args.gpus)

    if os.path.exists('malconv.h5'):
        print("restoring malconv.h5 from disk for continuation training...")
        from keras.models import load_model
        basemodel = load_model('malconv.h5')
        _, maxlen, embedding_size = basemodel.layers[1].output_shape
        input_dim
    else:
        maxlen = 2**20 # 1MB
        embedding_size = 8 
        

        # define model structure
        inp = Input( shape=(maxlen,))
        emb = Embedding( input_dim, embedding_size )( inp )
        filt = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='relu', padding='valid' )(emb)
        attn = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='sigmoid', padding='valid')(emb)
        gated = Multiply()([filt,attn])
        feat = GlobalMaxPooling1D()( gated )
        dense = Dense(128, activation='relu')(feat)
        outp = Dense(10, activation='softmax')(dense)

        basemodel = Model( inp, outp )

    basemodel.summary() 

    print("Using %i GPUs" %ngpus)

    if ngpus > 1:
        model = multi_gpu.make_parallel(basemodel,ngpus)
    else:
        model = basemodel

    from tensorflow.keras.optimizers import SGD
    model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3), metrics=[metrics.accuracy] )

    def bytez_to_numpy(bytez,maxlen):
        b = np.ones( (maxlen,), dtype=np.uint16 )*padding_char
        bytez = np.frombuffer( bytez[:maxlen], dtype=np.uint8 )
        b[:len(bytez)] = bytez
        return b

    def getfile_service(path,maxlen=maxlen):        
        with open(path,'rb') as f:
            return bytez_to_numpy( f.read(), maxlen )        

    def generator( hashes, labels, batch_size, shuffle=True ):
        X = []
        y = []
        zipped = list(zip(hashes, labels))
        while True:
            if shuffle:
                random.shuffle( zipped )
            for sha256,l in zipped:
                x = getfile_service(sha256)
                if x is None:
                    continue
                X.append( x )
                y.append( l )
                if len(X) == batch_size:
                    # print("X")
                    # print(X)
                    # print("y")
                    # print(np.asarray(y).shape)
                    yield np.asarray(X,dtype=np.uint16), to_categorical(np.asarray(y), 10)
                    X = []
                    y = []

    import pandas as pd
    # train_labels = pd.read_csv('ember_training.csv.gz')
    # train_labels = train_labels[ train_labels['y'] != -1 ] # get only labeled samples
    # labels = train_labels['y'].tolist()
    # hashes = train_labels['sha256'].tolist()

    from sklearn.model_selection import train_test_split
    # hashes_train, hashes_val, labels_train, labels_val = train_test_split( hashes, labels, test_size=200 )

    train_label_path = '/content/drive/MyDrive/microsoft_big'
    train_data_path = '/content/drive/MyDrive/microsoft_big/train'

    all_labels_path = os.path.join(train_label_path,'trainLabels.csv')
    all_files_path = train_data_path

    all_labels_df = pd.read_csv(all_labels_path)
    all_files = os.listdir(all_files_path)
    for i in range(len(all_files)):
        all_files[i] = all_files[i].split('.')[0]

    print(len(all_labels_df))


    print(all_labels_df)
    print(all_files)


    # remove the labels not in our limited trainings
    all_labels_df.drop(all_labels_df[~all_labels_df.Id.isin(all_files)].index, inplace=True)
    all_labels= all_labels_df.reset_index()
    print(len(all_labels_df))

    X= all_labels_df['Id']
    y= all_labels_df['Class']


    # using the train test split function
    X_train, X_val,y_train, y_val = train_test_split(X,y ,
                                    random_state=104, 
                                    test_size=0.25, 
                                    shuffle=True)
    X_train = X_train.to_list()
    y_train = y_train.to_list()
    X_val = X_val.to_list()
    y_val = y_val.to_list()

    for i in range(len(X_train)):
        X_train[i] = os.path.join(train_data_path, X_train[i]+'.bytes')

    for i in range(len(X_val)):
        X_val[i] = os.path.join(train_data_path, X_val[i]+'.bytes' )


    train_gen = generator( X_train, y_train, batch_size )
    val_gen = generator( X_val, y_val, batch_size )

    from keras.callbacks import LearningRateScheduler

    base = K.get_value( model.optimizer.lr )
    def schedule(epoch):
        return base / 10.0**(epoch//2)

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(X_train)//batch_size,
        epochs=10,
        validation_data=val_gen,
        callbacks=[ LearningRateScheduler( schedule ) ],
        # validation_steps=int(math.ceil(len(X_val)/batch_size)),
    )

    basemodel.save('malconv.h5')

    # test_labels = pd.read_csv('ember_test.csv.gz')
    # labels_test = test_labels['y'].tolist()
    # hashes_test = test_labels['sha256'].tolist()

    # test_generator = generator(hashes_test,labels_test,batch_size=1,shuffle=False)
    # test_p = basemodel.predict_generator( test_generator, steps=len(test_labels), verbose=1 )


if __name__ == '__main__':

    main() # uncomment this line after fixing the URL NotImplementedError above