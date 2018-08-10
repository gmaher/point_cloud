import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def get_batch(X,Y,N,n=32, y_index='all'):
    inds = np.random.choice(range(N),size=n, replace=False)
    x = X[inds,:,:,:]
    y = Y[inds,:,:,:]
    if y_index != 'all':
        y = y[:,:,:,:,y_index]
        y = y[:,:,:,:,np.newaxis]
    x = x[:,:,:,:,np.newaxis]
    return x,y

def conv3D(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32, init=1e-3, scope='conv3d', reuse=False):
    #batch,W,H,D,C
    with tf.variable_scope(scope,reuse=reuse):
        s = x.get_shape()
        shape = shape +[int(s[4]),nfilters]
        W = tf.Variable(tf.random_normal(shape=shape,stddev=init),
            name='W')
        b = tf.Variable(tf.ones([nfilters])*init, name='b')

        h = tf.nn.convolution(x,W,padding='SAME',strides=[1,1,1], name='h')+b

        a = activation(h)

        return a

def repeat(x,axis,repeat):
    s = x.get_shape().as_list()
    splits = tf.split(value=x,num_or_size_splits=s[axis],axis=axis)
    rep = [s for s in splits for _ in range(repeat)]
    return tf.concat(rep,axis)

def resize_tensor(x,scales=[1,2,2,2,1]):
    out = x
    for i in range(1,len(scales)):
        out = repeat(out,i,scales[i])
    return out

def upsample3D(x, scope='upsample'):
    with tf.variable_scope(scope):
        #o = resize_tensor(x)
        init = 1e-3
        s = x.get_shape().as_list()
        batch_size = tf.shape(x)[0]
        w_shape = [2,2,2,s[4],s[4]]
        o_shape = [batch_size,2*s[1],2*s[2],2*s[3],s[4]]
        print('upsample ', o_shape)
        strides = [1,2,2,2,1]
        W = tf.Variable(tf.random_normal(shape=w_shape,stddev=init),
            name='W_upsample')
        o = tf.nn.conv3d_transpose(x,W,o_shape,strides=strides,
            padding='SAME')
        return o

def conv3D_N(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32,init=1e-3,scope='conv3dn',N=2):
    o = x
    for i in range(N):
        s = scope +'_'+str(i)
        o = conv3D(o,activation,shape,nfilters,init,s)

    return o

def resnet_conv3D_N(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32,init=1e-3,scope='conv3dn',N=2):
    o = x
    for i in range(N-1):
        s = scope +'_'+str(i)
        o = conv3D(o,activation,shape,nfilters,init,s)

    s = scope +'_'+str(N)
    o = conv3D(o,tf.identity,shape,nfilters,init,s)
    y = activation(o+x)

    return y

def unetBlock(x,nfilters=32,scope='unet3d',init=1e-3,activation=tf.nn.relu):
    with tf.variable_scope(scope):
        o1 = conv3D(x,nfilters=nfilters,init=init,activation=activation)
        o2 = conv3D(o1,nfilters=2*nfilters,init=init,activation=activation)
        o3 = tf.nn.pool(o2,[2,2,2],strides=[2,2,2],pooling_type='MAX',padding='SAME',name='pool')
    return o3,o2

def unetUpsampleBlock(x,y,activation=tf.nn.relu,init=1e-3,nfilters=32,scope='unet3dupsample'):
    with tf.variable_scope(scope):
        o = upsample3D(y)
        o = tf.concat([o,x],axis=4)
        o = conv3D(o,nfilters=nfilters,init=init,activation=activation)
        o = conv3D(o,nfilters=nfilters,init=init,activation=activation)
        return o

def UNET3D(x,activation=tf.nn.relu,nfilters=32,init=1e-3):
    o1_down,o1 = unetBlock(x,nfilters=nfilters,scope='layer1',init=init,activation=activation)
    print( o1,o1_down)
    o2_down,o2 = unetBlock(o1_down,nfilters=2*nfilters,scope='layer2',init=init,activation=activation)
    print( o2,o2_down)
    o3_down,o3 = unetBlock(o2_down,nfilters=4*nfilters,scope='layer3',init=init,activation=activation)
    print( o3,o3_down)
    o4_down,o4 = unetBlock(o3_down,nfilters=8*nfilters,scope='layer4',init=init,activation=activation)
    print( o4,o4_down)

    a_3 = unetUpsampleBlock(o3,o4,nfilters=8*nfilters,scope='o_layer3',init=init,activation=activation)
    print( a_3)
    a_2 = unetUpsampleBlock(o2,a_3,nfilters=4*nfilters,scope='o_layer2',init=init,activation=activation)
    print( a_2)
    a_1 = unetUpsampleBlock(o1,a_2,nfilters=2*nfilters,scope='o_layer1',init=init,activation=activation)
    print( a_1)
    yhat = conv3D(a_1,tf.identity,shape=[1,1,1],nfilters=1,scope='yhat',init=init)
    print( yhat)
    yclass = tf.sigmoid(yhat)
    print( yclass)
    return yhat,yclass


def I2INetBlock(x,nfilters=32,init='xavier',activation=tf.nn.relu, num=2,scope='i2i'):
    with tf.variable_scope(scope):
        o1 = conv3D(x,nfilters=nfilters,init=init,activation=activation)
        o2 = conv3D(o1,nfilters=nfilters,init=init,activation=activation)
        if num==3:
            o2 = conv3D(o1,nfilters=nfilters,init=init,activation=activation)
        o3 = tf.nn.pool(o2,[2,2,2],strides=[2,2,2],pooling_type='AVG',padding='SAME',name='pool')
        print( 'i2inetdown ', o3.get_shape().as_list())
    return o3,o2

def I2INetUpsampleBlock(x,y,n1=64,n2=32,init='xavier',activation=tf.nn.relu,scope='ublock'):
    with tf.variable_scope(scope):
        print( "i2inet x ", x.get_shape().as_list())
        y = upsample3D(y, scope=scope)
        x = tf.concat([x,y],axis=4)
        o = conv3D(x,shape=[1,1,1],nfilters=n1,init=init,activation=activation)
        o1 = conv3D(o,nfilters=n2,init=init,activation=activation)
        o2 = conv3D(o1,nfilters=n2,init=init,activation=activation)
        print( 'i2inetup ', o2.get_shape().as_list())
        return o2

def I2INet(x, nfilters=32, init='xavier', activation=tf.nn.relu):

    d1,a1 = I2INetBlock(x,nfilters=nfilters,init=init,activation=activation,scope='block1')
    d2,a2 = I2INetBlock(d1,nfilters=4*nfilters,init=init,activation=activation, scope='block2')
    d3,a3 = I2INetBlock(d2,nfilters=8*nfilters,init=init,activation=activation, scope='block3')
    d4,a4 = I2INetBlock(d3,nfilters=16*nfilters,init=init,activation=activation, scope='block4')

    o4 = I2INetUpsampleBlock(a3,a4,n1=16*nfilters,n2=8*nfilters,init=init,activation=activation, scope='ublock1')
    o3 = I2INetUpsampleBlock(a2,o4,n1=8*nfilters,n2=4*nfilters,init=init,activation=activation, scope='ublock2')
    o2 = I2INetUpsampleBlock(a1,o3,n1=4*nfilters,n2=nfilters,init=init,activation=activation, scope='ublock3')

    out = conv3D(o2,nfilters=1,activation=tf.identity)
    out_class = tf.nn.sigmoid(out)

    return out_class,out,o3,o4
