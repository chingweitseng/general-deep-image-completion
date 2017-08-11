import tensorflow as tf
import numpy as np

# Build a class for model
class Model():
    def __init__(self):
        pass

    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):

            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)


            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed

    def resize_conv_layer(self, bottom, filter_shape, resize_scale=2, activation=tf.identity, padding='SAME', stride=1, name=None):
        width = bottom.get_shape().as_list()[1]
        height = bottom.get_shape().as_list()[2]
        bottom = tf.image.resize_nearest_neighbor(bottom, [width*resize_scale, height*resize_scale])
        bias = self.new_conv_layer(bottom, filter_shape, stride=1, name=name)
        return bias

    def build_reconstruction( self, images, is_train ):
      
        batch_size = images.get_shape().as_list()[0]
        
        with tf.variable_scope('GEN'):
      
            # Encoder: Using residual blocks for feature learning
            
            conv_feat = self.new_conv_layer(images, [4,4,3,32], stride=1, name="conv_feat" )

            # residual block 1
            conv1_start = self.new_conv_layer(conv_feat, [4,4,32,32], stride=1, name="conv1_start" )
            bn1_start = tf.nn.elu(conv1_start)
            #bn1_start = self.leaky_relu(self.batchnorm(conv1_start, is_train, name='bn1_start'))
            conv1 = self.new_conv_layer(bn1_start, [4,4,32,32], stride=1, name="conv1" )
            bn1 = tf.nn.elu(conv1)
            #bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            conv1_res = bn1 + conv_feat
            conv1_stride = self.new_conv_layer(conv1_res, [4,4,32,64], stride=2, name="conv1_stride") # downsample 1:128->64
            
            # residual block 2
            conv2_start = self.new_conv_layer(conv1_stride, [4,4,64,64], stride=1, name="conv2_start" )
            bn2_start = tf.nn.elu(conv2_start)
            #bn2_start = self.leaky_relu(self.batchnorm(conv2_start, is_train, name='bn2_start'))
            conv2 = self.new_conv_layer(bn2_start, [4,4,64,64], stride=1, name="conv2" )
            bn2 = tf.nn.elu(conv2)
            #bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            conv2_res = bn2 + conv1_stride
            conv2_stride = self.new_conv_layer(conv2_res, [4,4,64,128], stride=2, name="conv2_stride") # downsample 2:64->32
            
            # residual block 3
            conv3_start = self.new_conv_layer(conv2_stride, [4,4,128,128], stride=1, name="conv3_start" )
            bn3_start = tf.nn.elu(conv3_start)
            #bn3_start = self.leaky_relu(self.batchnorm(conv3_start, is_train, name='bn3_start'))
            conv3 = self.new_conv_layer(bn3_start, [4,4,128,128], stride=1, name="conv3" )
            bn3 = tf.nn.elu(conv3)
            #bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            conv3_res = bn3 + conv2_stride
            conv3_stride = self.new_conv_layer(conv3_res, [4,4,128,256], stride=2, name="conv3_stride") # downsample 3:32->16
            
            # residual block 4
            conv4_start = self.new_conv_layer(conv3_stride, [4,4,256,256], stride=1, name="conv4_start" )
            bn4_start = tf.nn.elu(conv4_start)
            #bn4_start = self.leaky_relu(self.batchnorm(conv4_start, is_train, name='bn4_start'))
            conv4 = self.new_conv_layer(bn4_start, [4,4,256,256], stride=1, name="conv4" )
            bn4 = tf.nn.elu(conv4)
            #bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))
            conv4_res = bn4 + conv3_stride
            conv4_stride = self.new_conv_layer(conv4_res, [4,4,256,512], stride=2, name="conv4_stride") # downsample 4:16->8
            
            # residual block 5
            conv5_start = self.new_conv_layer(conv4_stride, [4,4,512,512], stride=1, name="conv5_start" )
            bn5_start = tf.nn.elu(conv5_start)
            #bn5_start = self.leaky_relu(self.batchnorm(conv5_start, is_train, name='bn5_start'))
            conv5 = self.new_conv_layer(bn5_start, [4,4,512,512], stride=1, name="conv5" )
            bn5 = tf.nn.elu(conv5)
            #bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5'))
            conv5_res = bn5 + conv4_stride
            
            # Feature patches
            conv5_stride = self.new_conv_layer(conv5_res, [4,4,512,512], stride=2, name="conv5_stride") # downsample 4:8->4

            # Decoder: Reconstruction
            
            # 5
            deconv5_fs = self.resize_conv_layer(conv5_stride, [4,4,512,512], resize_scale=2, stride=1, name="deconv5_fs")
            
            deconv5 = self.new_conv_layer(deconv5_fs, [4,4,512, 512], stride=1, name="deconv5" )
            debn5 = tf.nn.elu(deconv5)
            deconv5_1 = self.new_conv_layer(debn5, [4,4,512, 512], stride=1, name="deconv5_1" )
            debn5_1 = tf.nn.elu(deconv5_1)
            deconv5_res = deconv5_fs + debn5_1
            
            deconv5_res = tf.concat([deconv5_res, conv4_stride], 3)
            channels5 = deconv5_res.get_shape().as_list()[3]
            
            # 4
            deconv4_fs = self.resize_conv_layer( deconv5_res, [4,4,channels5,256], resize_scale=2, stride=1, name="deconv4_fs")
            
            deconv4 = self.new_conv_layer(deconv4_fs, [4,4,256, 256], stride=1, name="deconv4" )
            debn4 = tf.nn.elu(deconv4)
            deconv4_1 = self.new_conv_layer(debn4, [4,4,256, 256], stride=1, name="deconv4_1" )
            debn4_1 = tf.nn.elu(deconv4_1)
            deconv4_res = deconv4_fs + debn4_1
            
            deconv4_res = tf.concat([deconv4_res, conv3_stride], 3)
            channels4 = deconv4_res.get_shape().as_list()[3]
            
            # 3
            deconv3_fs = self.resize_conv_layer( deconv4_res, [4,4,channels4,128], resize_scale=2, stride=1, name="deconv3_fs")
            
            deconv3 = self.new_conv_layer(deconv3_fs, [4,4,128, 128], stride=1, name="deconv3" )
            debn3 = tf.nn.elu(deconv3)
            deconv3_1 = self.new_conv_layer(debn3, [4,4,128, 128], stride=1, name="deconv3_1" )
            debn3_1 = tf.nn.elu(deconv3_1)
            deconv3_res = deconv3_fs + debn3_1
            
            deconv3_res = tf.concat([deconv3_res, conv2_stride], 3)
            channels3 = deconv3_res.get_shape().as_list()[3]
            
            
            # 2
            deconv2_fs = self.resize_conv_layer( deconv3_res, [4,4,channels3,64], resize_scale=2, stride=1, name="deconv2_fs")
            
            deconv2 = self.new_conv_layer(deconv2_fs, [4,4,64, 64], stride=1, name="deconv2" )
            debn2 = tf.nn.elu(deconv2)

            deconv2_1 = self.new_conv_layer(debn2, [4,4,64, 64], stride=1, name="deconv2_1" )
            debn2_1 = tf.nn.elu(deconv2_1)

            deconv2_res = deconv2_fs + debn2_1
            
            deconv2_res = tf.concat([deconv2_res, conv1_stride], 3)
            channels2 = deconv2_res.get_shape().as_list()[3]
            
            # 1
            deconv1_fs = self.resize_conv_layer( deconv2_res, [4,4,channels2,32], resize_scale=2, stride=1, name="deconv1_fs")
            
            deconv1 = self.new_conv_layer(deconv1_fs, [4,4,32, 32], stride=1, name="deconv1" )
            debn1 = tf.nn.elu(deconv1)

            deconv1_1 = self.new_conv_layer(debn1, [4,4,32, 32], stride=1, name="deconv1_1" )
            debn1_1 = tf.nn.elu(deconv1_1)

            deconv1_res = deconv1_fs + debn1_1
            
            deconv1_res = tf.concat([deconv1_res, conv_feat], 3)
            channels1 = deconv1_res.get_shape().as_list()[3]
            
            # Output
            recon = self.new_conv_layer(deconv1_res, [4,4,channels1, 3], stride=1, name="recon" )

        return recon

    def build_adversarial(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
      
            ## conv_start = self.new_conv_layer(images, [4,4,3,64], stride=1, name="conv_start" )
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))

            output = self.new_fc_layer( bn4, output_size=1, name='output')

        return output[:,0]
  
  