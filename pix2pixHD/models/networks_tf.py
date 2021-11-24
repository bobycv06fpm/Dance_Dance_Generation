

import tensorflow as tf

def LocalEnhancer(input_Gen,input_nc, output_nc, nf=32, n_downsample_global=3, n_blocks_global=9, n_blocks_local=3, norm_layer='', padding_type='reflect'):

        initializer = tf.random_normal_initializer(0., 0.02)
        
        ############################################ call global generator model ######################################## 
               
        nf_global = nf * 2
        model_global = GlobalGenerator(input_Gen, output_nc, nf_global, n_downsample_global, n_blocks_global)       
        model_global = model_global.get_layer("last_layer").output# get output of final convolution layers        
                   
        ############################################# local enhancer layers ##############################################
        
        ### downsample 
        downsample_G2 = downsample(input_Gen, nf, kernel_size=7, padding_type = "REFLECT")   
        downsample_G2 = downsample(downsample_G2, nf * 2, kernel_size=3, stride=2, padding_type = "zero")
        
        ####concatenat feature map of last layer of Global Enhancer with featurmap of downsample layer of enhancer network
        resnet_input = tf.keras.layers.concatenate(downsample_G2, model_global)
        ### residual blocks 
        residual_block_G2 = ResnetBlock(resnet_input, nf * 2, padding_type="REFLECT", n_blocks_local)     
        ### upsample
        upsample_G2 = ConvTranspos(residual_block_G2, nf, kernel_size = 3, stride = 2, padding_type = "zero", output_padding = 1)       
        ### final convolution 
 
        output= final_layer(upsample_G2, output_nc, kernel_size = 7, padding_type = "REFLECT")
        
        return tf.keras.Model(input= input_Gen, outputs=output)

        
########################################Global enhancer layers ##############################################
        
def GlobalGenerator(input_Gen, output_nc, nf=64, n_downsampling=3, n_blocks=9):
    
    initializer = tf.random_normal_initializer(0., 0.02)
    paddings = tf.constant([[3, 3,], [3, 3]])
    ####resize input 2048 8 1024 to 1024*512
    input_Gen = tf.keras.layers.AveragePooling2D(pool_size= 3, stride=2)(input_Gen)    
    
    #####downsample
    downsample_G1 = downsample(input_globalG, nf, kernel_size=7, padding=0, padding_type = "REFLECT")
    
    for i in range(n_downsampling):
            mult = 2**i
            downsample_G1 = downsample(downsample_G1, nf * mult * 2, kernel_size=3, stride=2, padding_type = "zero")

    ### resnet blocks
    mult = 2**n_downsampling
    ResnetBlock_G1= ResnetBlock(downsample_G1, nf * mult, padding_type="REFLECT", n_blocks)
        
    ### upsample
    mult = 2**(n_downsampling)
    upsample_G1 =  ConvTranspos(ResnetBlock_G1, int(nf * mult / 2), kernel_size=3, stride=2, padding_type = "zero", output_padding=1)      
    for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            upsample_G1 = ConvTranspos(upsample_G1, int(nf * mult / 2), kernel_size=3, stride=2, padding_type = "zero", output_padding=1)
            
    ####final layer(output)                  
    output= final_layer(input_last,output_nc, kernel_size = 7, padding_type = "REFLECT")     
    return tf.keras.Model(inputs= input_Gen, outputs=output)

d

def downsample(input_down, nf, kernel_size, stride, padding_type=None):
      if  padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           input_down = tf.pad(input_down, paddings, "REFLECT")
      elif padding_type == "zero" :
           input_down = tf.keras.layers.ZeroPadding2D(padding =(1,1))(input_down)
      else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
      downsample = tf.keras.layers.Conv2D(nf, kernel_size=7)(input_down)
      downsample = tf.contrib.layers.instance_norm(downsample)
      downsample = tf.keras.layers.ReLU()(downsample)
      return downsample

def ResnetBlock(input_resnet, nf, padding_type=None, use_dropout=False, n_block):
    for i in (n_block):
       if  padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           input_resnet = tf.pad(input_resnet, paddings, "REFLECT")
       elif padding_type == "zero" :
           input_resnet = tf.keras.layers.ZeroPadding2D(padding =(1,1))(input_resnet)
       else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
       resnet = tf.keras.layers.Conv2D(nf, kernel_size=3)(input_resnet)
       resnet = tf.contrib.layers.instance_norm(resnet)
       resnet = tf.keras.layers.ReLU()(resnet)
       if use_dropout:
           resnet = tf.keras.layers.Dropout(0.5)(resnet)
           
       if  padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           input_resnet = tf.pad(resnet, paddings, "REFLECT")
       elif padding_type == "zero" :
           input_resnet = tf.keras.layers.ZeroPadding2D(padding =(1,1))(resnet)   
       else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
       resnet = tf.keras.layers.Conv2D(nf, kernel_size=3)(input_resnet)
       resnet = tf.contrib.layers.instance_norm(resnet)
    return resnet



def ConvTranspos(input_transpos,nf, kernel_size, stride, padding_type = None, output_padding=None):
    
    if padding_type == "zero" :
           input_resnet = tf.keras.layers.ZeroPadding2D(padding =(1,1))(input_transpos)   
       else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    Transpos = tf.keras.layers.Conv2DTranspose(nf, kernel_size=kernel_size, stride=stride, output_padding=output_padding)(input_transpos)
    return Transpos
 output= final_layer(input_last,output_nc, kernel_size = 7, padding_type = "REFLECT")    
def final_layer(input_last, output_nc, kernel_size, padding_type=None):
     if padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           output = tf.pad(input_last, paddings, "REFLECT")
     else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    output = tf.keras.layers.Conv2D(output_nc, kernel_size=7)(output)
    output = tf.keras.activations.tanh(output)
    return output
    
    
    
    
    
    
    
   