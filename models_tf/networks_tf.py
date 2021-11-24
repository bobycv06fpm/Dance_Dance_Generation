

import tensorflow as tf

################################################################################################
#                                        ##Genarator##                                         #
################################################################################################ 
def LocalEnhancer(output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, n_blocks_local=3):
#########################################################################################################################################################################################################  
#input:                                                                                                                                                                                                 #
#ouput_nc = # of output channel                                                                                                                                                                         #
# ngf = # fo filter in generator                                                                                                                                                                        #
# n_downsample_global = # of downsample block in golbal generator(G1)                                                                                                                                   #  
# n_blocks_global = # of resnet block in golbal generator (G1)                                                                                                                                          #
# n_blocks_local = # of resnet block in local enhancer (G2)                                                                                                                                             #
#output:                                                                                                                                                                                                # 
#local enhancer model(first is trained the global generator and then  trained  the  local  enhancer  in  the  order  of  their  reso-lutions.We  then  jointly  fine-tune  all  the  networks  together)#    
#########################################################################################################################################################################################################   
        
        input_Gen = tf.keras.layers.Input(shape=[None,None,None])
        ############################################ call global generator model #########################################
        ngf_global = ngf * 2
        model_global = GlobalGenerator(input_Gen, output_nc, ngf_global, n_downsample_global, n_blocks_global)
        model2 = tf.keras.models.Model(inputs=input_Gen, outputs=modle_global)
        # TODO: training
        ############################################# local enhancer layers ##############################################
        ### downsample 
        downsample_G2 = conv_layer(input_Gen, ngf, ksize=7, padding_type ="REFLECT", activation='ReLU') 
        downsample_G2 = conv_layer(downsample_G2, ngf * 2, ksize=3, stride=2, padding_type ="zero", activation='ReLU', pad=1)
        
        ### element-wise sum of two feature maps: the output feature map of downsample_G2, and the last feature map of the back-end of the global generatornetwork
        resnet_G2 = downsample_G2 + model2.layers[-1]
        
        ### residual blocks ??????
        for i in range(n_blocks_local):
                resnet_G2 = ResnetBlock(resnet_G2, ngf * 2, paddingtype="REFLECT")
        
        ### upsample
        upsample_G2 = ConvTranspos(resnet_G2, ngf, kernel_size=3, stride=2, padding_type ="zero", output_padding=1) 
        
        ### final convolution 
        output= final_layer(upsample_G2, output_nc, kernel_size=7, padding_type ="REFLECT")
        
        return tf.keras.Model(inputs= input_Gen, outputs=output)

        
########################################Global enhancer layers ##############################################
        
def GlobalGenerator(input_Gen, output_nc, ngf=64, n_downsampling=4, n_blocks=9):
    

    ####resize input 2048*1024 to 1024*512
    input_Gen = tf.keras.layers.AveragePooling2D(pool_size=3, stride=2, padding="same")(input_Gen)      
    downsample_G1 = conv_layer(input_Gen, ngf, ksize=7,padding_type = "REFLECT", activation='ReLU')
    #####downsample 
    for i in range(n_downsampling):
            mult = 2**i
            downsample_G1 = conv_layer(downsample_G1, ngf * mult * 2, ksize=3, stride=2, padding_type="zero", activation='ReLU', pad=1)

    ### resnet blocks
    mult = 2**(n_downsampling)
    ResnetBlock_G1= ResnetBlock(downsample_G1, ngf * mult, paddingtype="REFLECT")
    for i in range(1, n_blocks):
        ResnetBlock_G1= ResnetBlock(ResnetBlock_G1, ngf * mult, paddingtype="REFLECT")
        
    ### upsample
    mult = 2**(n_downsampling)
    upsample_G1 =  ConvTranspos(ResnetBlock_G1, int(ngf * mult / 2), kernel_size=3, stride=2, padding_type ="zero", output_padding=1)      
    for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            upsample_G1 = ConvTranspos(upsample_G1, int(ngf * mult / 2), kernel_size=3, stride=2, padding_type ="zero", output_padding=1)
            
    ####final layer(output)                  
    output= final_layer(upsample_G1, output_nc, kernel_size=7, padding_type ="REFLECT")  
    
    return output

########Convolution layer(conv-norm-activation)
def Conv_layer(input_conv, ngf, ksize, stride=1, padding_type=None, activation=None, ns=1, pad=0, no_norm=False):
    
      initializer = tf.random_normal_initializer(0., 0.02)
      if  padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           input_conv = tf.pad(input_conv, paddings, "REFLECT")
      elif padding_type == "zero" :
           input_conv = tf.keras.layers.ZeroPadding2D(padding=pad)(input_conv)
      else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
      if activation=='ReLU':
           activation = tf.keras.layers.ReLU()
           if no_norm:
                conv_layer = tf.keras.layers.Conv2D(filters=ngf, kernel_size=ksize, strides=stride, kernel_initializer=initializer)(input_conv)
                conv_layer = activation(conv_layer)
                return conv_layer
      elif activation=='LeakyReLU':
           activation = tf.keras.layers.LeakyReLU(ns)
           if no_norm:
                conv_layer = tf.keras.layers.Conv2D(filters=ngf, kernel_size=ksize, strides=stride, kernel_initializer=initializer)(input_conv)
                conv_layer = activation(conv_layer)
                return conv_layer
  
      else:
          conv_layer = tf.keras.layers.Conv2D(filters=ngf, kernel_size=ksize, strides=stride, kernel_initializer=initializer)(input_conv)
          if no_norm:
              return conv_layer
          
          conv_layer = tf.contrib.layers.instance_norm(conv_layer)
          return conv_layer
                      
      conv_layer = tf.keras.layers.Conv2D(filters=ngf, kernel_size=ksize, strides=stride, kernel_initializer=initializer)(input_conv)
      conv_layer = tf.contrib.layers.instance_norm(conv_layer)
      conv_layer = activation(conv_layer)
      return conv_layer
  
############ResnetBlocK(conv2D-instance normalization-Relu) 
def ResnetBlock(input_resnet, ngf, paddingtype=None, use_dropout=False, pad_res=0):
  
       resnet = Conv_layer(input_resnet, ngf, ksize=3, padding_type=paddingtype, activation='ReLU', pad=pad_res)
       if use_dropout:
           resnet = tf.keras.layers.Dropout(0.5)(resnet)
           
       resnet = Conv_layer(resnet, ngf, ksize=3, padding_type=paddingtype, pad=pad_res)
       return input_resnet + resnet

###############Upsampling(conv2Dtranspos-instanc norm-Relu)
def ConvTranspos(input_resnet, ngf, kernel_size, stride, padding_type = None, output_padding=None):
    
    initializer = tf.random_normal_initializer(0., 0.02)
    if padding_type == "zero" :
           input_resnet = tf.keras.layers.ZeroPadding2D(padding =(1,1))(input_transpos)   
       else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
    Transpos = tf.keras.layers.Conv2DTranspose(ngf, kernel_size=kernel_size, stride=stride, output_padding=output_padding, kernel_initializer=initializer)(input_transpos)
    Transpos = tf.contrib.layers.instance_norm(Transpos)
    Transpos = tf.keras.layers.ReLU()(Transpos)
    return Transpos

#################ouputlaye(conv2D-tanh activation)     
def final_layer(output_nc, kernel_size, padding_type=None):
    
     initializer = tf.random_normal_initializer(0., 0.02)
     if padding_type == "REFLECT" :   
           padding = tf.constant([[3, 3,], [3, 3]])
           output = tf.pad(input_last, paddings, "REFLECT")
     else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    output = tf.keras.layers.Conv2D(output_nc, kernel_size=kernel_size, kernel_initializer=initializer)(output)
    output = tf.keras.activations.tanh(output)
    return output 


    
    
##################################################################################################################
#                                          ###Discrimenator###                                                   # 
##################################################################################################################

def MultiscaleDiscriminator(ndf=64, n_layers=3, use_sigmoid=False, num_D=3):
    
    input_D = tf.keras.layers.Input(shape=[None,None,None])
    Avg_pool = tf.keras.layers.AveragePooling2D(pool_size=3, stride=2, padding="same") 
    D1 = NLayerDiscriminator(input_D, ndf=64, n_layers=3, use_sigmoid=False)
    model1 = tf.keras.models.Model(inputs=input_D, outputs=D1)
    
    input_D2 = Avg_pool(input_D)
    D2 = NLayerDiscriminator(input_D2, ndf=64, n_layers=3, use_sigmoid=False)
    model2 = tf.keras.models.Model(inputs=input_D2, outputs=D2)
    
    input_D3 = Avg_pool(input_D2)
    D3 = NLayerDiscriminator(input_D3, ndf=64, n_layers=3, use_sigmoid=False)
    model3 = tf.keras.models.Model(inputs=input_D3, outputs=D3)
    


def NLayerDiscriminator(input_D, ndf=64, n_layers=3, use_sigmoid=False):
    
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        output_ = Conv_layer(input_D, ndf, ksize=kw, stride=2, padding_type='zero', activation='LeakyReLU', ns=0.2, pad=padw, no_norm=True)
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            output_ = Conv_layer(output_, nf, ksize=kw, stride=2, padding_type='zero', activation='LeakyReLU', ns=0.2, pad=padw)

        nf_prev = nf
        nf = min(nf * 2, 512)
        output_ = Conv_layer(output_, nf, ksize=kw, stride=1, padding_type='zero', activation='LeakyReLU',ns=0.2, pad=padw)
        output_ = Conv_layer(output_, 1, ksize=kw, stride=1, padding_type='zero', pad=padw, no_norm=True)
        
        if use_sigmoid:
            output_ = tf.keras.activations.sigmoid(output_)
            
        return output_

                        
    
    
    
    
    
   