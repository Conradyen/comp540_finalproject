from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

'''
model-1
model from bio image segamentation
'''
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def u_net_model(X_train, Y_train,base_num_mask =16 ,act = 'relu',batch_size=16,epochs=100,validation_split=0.1,IMG_WIDTH = 256,IMG_HEIGHT = 256,IMG_CHANNELS = 3):

    print("initializing U Net model.......")

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(base_num_mask, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(base_num_mask, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(base_num_mask*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(base_num_mask*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(base_num_mask*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(base_num_mask*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(base_num_mask*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(base_num_mask*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(base_num_mask*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(base_num_mask*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(base_num_mask*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(base_num_mask*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(base_num_mask*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(base_num_mask*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(base_num_mask*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(base_num_mask*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(base_num_mask*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(base_num_mask*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(base_num_mask*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(base_num_mask, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(base_num_mask, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(base_num_mask, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-finalproject.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs,callbacks=[earlystopper, checkpointer])

    return model,redults
