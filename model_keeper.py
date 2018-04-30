def base_cnn(input_shape, c_drp_rate = 0.1, f_drp_rate = 0.2):
    # build Neural networks
    # for training a bad performance (model check)
    img_w, img_h, img_c = input_shape

    input_img = Input(shape= (img_w, img_h, img_c))

    x = Conv2D(128, (3,3), kernel_regularizer=l2(0.00001) )(input_img)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(5,5))(x)
    x = Dropout(c_drp_rate)(x)
    
    x = Conv2D(1024, (3,3), kernel_regularizer=l2(0.00001))(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(5,5))(x)
    x = Dropout(c_drp_rate)(x)
    """
    x = Conv2D(128, (3,3), kernel_regularizer=l2(0.00001))(x)
    x = Activation(activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2))(x)
    x = Dropout(c_drp_rate)(x)
    """
    x = GlobalAveragePooling2D()(x)
    
    """
    x = Dense(128, activation='relu')(x)
    x = Dropout(f_drp_rate)(x)
    """
    
    out = Dense(2, activation='softmax')(x)
    model = Model(inputs= [input_img], outputs=[out])
    
    return model