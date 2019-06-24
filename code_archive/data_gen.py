def train_data_gen(split):
    from keras_preprocessing.image import ImageDataGenerator
    import pandas as pd
    import math
    df=pd.read_csv('/home/ubuntu/derm_assist/data/labels/train_df.csv')
    columns=["MEL"]
    num_images = df.shape[0]
    split_index = math.floor(split*num_images)

    datagen=ImageDataGenerator(rescale=1./255.)                      

    train_generator=datagen.flow_from_dataframe(
    dataframe=df[:split_index],
    directory='/home/ubuntu/derm_assist/data/ISIC_2019_Training_Input',
    #directory=r'C:\Users\VAMS_2\Dropbox\ML\Insight\derm_assist\data\ISIC_2019_Training_Input',
    x_col="image",
    #has_ext=False,
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(224,224))

    return train_generator

def val_data_gen(split):
    from keras_preprocessing.image import ImageDataGenerator
    import pandas as pd
    import math
    df=pd.read_csv('/home/ubuntu/derm_assist/data/labels/train_df.csv')
    columns=["MEL"]
    num_images = df.shape[0]
    split_index = math.floor(split*num_images)

    datagen=ImageDataGenerator(rescale=1./255.)                      

    val_generator=datagen.flow_from_dataframe(
    dataframe=df[split_index:],
    directory='/home/ubuntu/derm_assist/data/ISIC_2019_Training_Input',
    #directory=r'C:\Users\VAMS_2\Dropbox\ML\Insight\derm_assist\data\ISIC_2019_Training_Input',
    x_col="image",
    #has_ext=False,
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(224,224))

    return val_generator





