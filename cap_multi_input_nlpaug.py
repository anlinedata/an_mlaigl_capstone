# -*- coding: utf-8 -*-
"""CAP_multi_Input_nlpaug_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PM_l0ahQLTXwtnoRHUf3ifZzlhyEGqAk
"""


def processModel():
    #pip install nlpaug

    # Commented out IPython magic to ensure Python compatibility.
    import string
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    # %matplotlib inline
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from scipy.stats import chi2_contingency
    from keras.utils import np_utils

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import TextVectorization
    from tensorflow.keras import layers, Input, Model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences # Preprocessing
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, GlobalMaxPool1D, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    import nlpaug
    import nlpaug.augmenter.word as naw

    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4') 

    data_path = r''

    #from google.colab import drive
    #drive.mount('/content/drive/')

    data_path = r''
    file_name = 'IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv'

    accident_df_raw = pd.read_csv(data_path + file_name)
    accident_df_raw.head()

    accident_df_raw = accident_df_raw.drop(['Unnamed: 0','Data'], axis=1)

    countries_encoder = LabelEncoder()
    accident_df_raw['Countries'] = countries_encoder.fit_transform(accident_df_raw['Countries'])

    local_encoder = LabelEncoder()
    accident_df_raw['Local'] = local_encoder.fit_transform(accident_df_raw['Local'])

    industry_sector_encoder = LabelEncoder()
    accident_df_raw['Industry Sector'] = industry_sector_encoder.fit_transform(accident_df_raw['Industry Sector'])

    gender_encoder = LabelEncoder()
    accident_df_raw['Genre'] = gender_encoder.fit_transform(accident_df_raw['Genre'])

    tp_encoder = LabelEncoder()
    accident_df_raw['Employee or Third Party'] = tp_encoder.fit_transform(accident_df_raw['Employee or Third Party'])

    risk_encoder = LabelEncoder()
    accident_df_raw['Critical Risk'] = risk_encoder.fit_transform(accident_df_raw['Critical Risk'])

    acident_level_encoder = LabelEncoder()
    accident_df_raw['Accident Level'] = acident_level_encoder.fit_transform(accident_df_raw['Accident Level'])

    potentital_accident_level_encoder = LabelEncoder()
    accident_df_raw['Potential Accident Level'] = potentital_accident_level_encoder.fit_transform(accident_df_raw['Potential Accident Level'])
    accident_df_raw.head()

    accident_df_raw.head()

    #accident_df_raw.info()

    acident_level_encoder.classes_

    #makes count plot and prints value counts
    def make_count_plot(data, x, hue=" "):
        if(len(hue) <= 1):
            sns.countplot(x=x,data=data)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
            sns.countplot(x=x,data=data, ax=axes[0])
            sns.countplot(x=x,data=data,hue=hue, ax=axes[1])
        plt.show()
        
        print(data[x].value_counts())

    #make_count_plot(accident_df_raw, 'Accident Level')

    #make_count_plot(accident_df_raw, 'Potential Accident Level')

    # from collections import Counter
    # from imblearn.over_sampling import RandomOverSampler
    # #Split train-test data

    # ros = RandomOverSampler()
    # # resampling X, y
    # X_ros, y_ros = ros.fit_resample(X, Y)
    # X_ros.shape, y_ros.shape





    non_text_imp_columns = ["Countries","Local","Industry Sector","Genre","Employee or Third Party","Critical Risk"]



    X_description = accident_df_raw["Description"].to_numpy()
    X_description.shape

    X_other_cols = accident_df_raw[non_text_imp_columns].to_numpy().astype(np.float32)
    X_other_cols.shape



    accident_df_raw.columns

    import math
    def create_augmented_records(level, upsample_size):
        level_records = accident_df_raw[accident_df_raw['Accident Level'] == level]
        len_records = len(level_records)
        level_records = level_records.reset_index()  # make sure indexes pair with number of rows

        num_sent = int(math.ceil(upsample_size / len_records))
        aug = naw.SynonymAug(aug_src='wordnet',aug_max=num_sent)

        level_records_augmented = []
        for index, row in level_records.iterrows():
            augmented_descriptions = aug.augment(row['Description'],n= num_sent)
            for augmented_description in augmented_descriptions:
                new_row = row.copy()
                new_row['Description'] = augmented_description
                level_records_augmented.append(new_row)
        return level_records_augmented[:upsample_size]

    upsample_size = 200
    level = 2
    level_0_downsampled_rows = accident_df_raw[accident_df_raw['Accident Level'] == 0].sample(upsample_size)
    level_1_augmented_records = create_augmented_records(1, upsample_size)
    level_2_augmented_records = create_augmented_records(2, upsample_size)
    level_3_augmented_records = create_augmented_records(3, upsample_size)
    level_4_augmented_records = create_augmented_records(4, upsample_size)

    augmented_accident_df = pd.DataFrame(level_0_downsampled_rows, columns=accident_df_raw.columns)
    augmented_accident_df = augmented_accident_df.append(level_1_augmented_records)
    augmented_accident_df = augmented_accident_df.append(level_2_augmented_records)
    augmented_accident_df = augmented_accident_df.append(level_3_augmented_records)
    augmented_accident_df = augmented_accident_df.append(level_4_augmented_records)
    augmented_accident_df.shape

    #make_count_plot(augmented_accident_df, 'Accident Level')

    #make_count_plot(augmented_accident_df, 'Potential Accident Level')

    X = augmented_accident_df[["Description", "Countries","Local","Industry Sector","Genre","Employee or Third Party","Critical Risk"]].to_numpy()
    accident_level_uniques, accident_level_ids = np.unique(augmented_accident_df['Accident Level'], return_inverse=True)
    potential_accident_level_uniques, potential_accident_level_ids = np.unique(augmented_accident_df['Potential Accident Level'], return_inverse=True)

    Y_accident_level = np_utils.to_categorical(accident_level_ids, len(accident_level_uniques))
    Y_potential_accident_level = np_utils.to_categorical(potential_accident_level_ids, len(potential_accident_level_uniques))

    X.shape, Y_accident_level.shape, Y_potential_accident_level.shape

    Y = np.concatenate((Y_accident_level, Y_potential_accident_level), axis=1)



    from sklearn.model_selection import train_test_split
    num_accident_level_classes = len(accident_level_uniques)
    num_potential_accident_level_classes = len(potential_accident_level_uniques)

    # Use train_test_split to split training data into training and validation sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.1, # dedicate 10% of samples to validation set
                                                        random_state=42) # random state for reproducibility





    X_train[:,0].shape, X_train[:,1:].shape

    X_desc_train = X_train[:,0]
    X_other_train = np.asarray(X_train[:,1:]).astype(np.float32)
    X_desc_train.shape,X_other_train.shape

    X_desc_test = X_test[:,0]
    X_other_test = np.asarray(X_test[:,1:]).astype(np.float32)
    X_desc_test.shape, X_other_test.shape

    # Fit the text vectorizer to the training text
    text_vectorizer = TextVectorization()
    text_vectorizer.adapt(X_description)

    import random
    # Choose a random sentence from the training dataset and tokenize it
    random_sentence = random.choice(X_description)
    #print(f"Original text:\n{random_sentence}\
    #      \n\nVectorized version:")
    text_vectorizer([random_sentence])

    # Get the unique words in the vocabulary
    words_in_vocab = text_vectorizer.get_vocabulary()
    top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
    bottom_5_words = words_in_vocab[-5:] # least common tokens
    #print(f"Number of words in vocab: {len(words_in_vocab)}")
    #print(f"Top 5 most common words: {top_5_words}") 
    #print(f"Bottom 5 least common words: {bottom_5_words}")

    max_vocab_length = len(words_in_vocab)
    max_length = round(max([len(i.split()) for i in X_description]))

    tf.random.set_seed(42)
    embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                                output_dim=128, # set size of embedding vector
                                embeddings_initializer="uniform", # default, intialize randomly
                                input_length=max_length, # how long is each input
                                name="embedding_1") 

    embedding

    other_features_input = Input(
        shape=(6,), name="other_features"
    )

    # Build model with the Functional API
    description_inputs = layers.Input(shape=(1,), dtype="string", name="description") # inputs are 1-dimensional strings
    desc_features = text_vectorizer(description_inputs) # turn the input text into numbers
    desc_features = embedding(desc_features) # create an embedding of the numerized numbers
    desc_features = layers.GlobalAveragePooling1D()(desc_features) # lower the dimensionality of the embedding

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([desc_features, other_features_input])
    # x = layers.concatenate([desc_features])

    num_accident_level_classes = 5
    accident_level = layers.Dense(num_accident_level_classes, activation="softmax")(x) # create the output layer, want binary outputs so use softmax activation
    potential_accident_level = layers.Dense(num_potential_accident_level_classes, activation="softmax")(x) # create the output layer, want binary outputs so use softmax activation
    #outputs = layers.Dense(num_classes, activation="softmax")(x) # create the output layer, want binary outputs so use softmax activation

    model = Model(
        inputs=[description_inputs, other_features_input],
        outputs=[accident_level, potential_accident_level],
    )

    #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    #model.summary()

    # Compile model
    model.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])


    batch_size = 32
    custom_early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min',
        restore_best_weights=True
    )

    Y_test[:,:num_accident_level_classes].shape

    Y_train[:,:num_accident_level_classes].shape, Y_train[:,num_accident_level_classes:num_accident_level_classes + num_potential_accident_level_classes].shape

    Y_test[:,:num_accident_level_classes].shape, Y_test[:,num_accident_level_classes:num_accident_level_classes + num_potential_accident_level_classes].shape

    Y_train_accident_level =  Y_train[:,:num_accident_level_classes]
    Y_train_potential_accident_level = Y_train[:,num_accident_level_classes:num_accident_level_classes + num_potential_accident_level_classes]

    Y_test_accident_level = Y_test[:,:num_accident_level_classes]
    Y_test_potential_accident_level = Y_test[:,num_accident_level_classes:num_accident_level_classes + num_potential_accident_level_classes]

    batch_size = 32
    custom_early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min',
        restore_best_weights=True
    )

    # Fit model
    # model_history = model.fit({"description": X_desc_train, "other_features": x_other_train_tensor},
    #                              Y_train,
    #                              epochs=50,
    #                              validation_data=({"description": X_desc_test, "other_features": x_other_test_tensor}, Y_test),
    #                              callbacks=[custom_early_stopping])

    # Fit model





    model_history = model.fit([ X_desc_train, X_other_train],
                                [Y_train_accident_level, Y_train_potential_accident_level],
                                epochs=50,
                                validation_data=([X_desc_test, X_other_test], [Y_test_accident_level, Y_test_potential_accident_level]),
                                callbacks=[custom_early_stopping])

    model.evaluate([X_desc_test, X_other_test], [Y_test_accident_level, Y_test_potential_accident_level])

    Y_pred = model.predict([X_desc_test, X_other_test])

    Y_test_accident_level.shape, Y_pred[0].shape

    Y_test_potential_accident_level.shape, Y_pred[1].shape

    Y_accident_level_test_decoded = accident_level_uniques[np.argmax(Y_test_accident_level, 1)]
    Y_accident_level_pred_decoded = accident_level_uniques[np.argmax(Y_pred[0], 1)]

    # Classification Report
    accident_level_cr = classification_report(Y_accident_level_test_decoded, Y_accident_level_pred_decoded)
    #print("Accident Level Classification Report:- ")
    #print(accident_level_cr)

    Y_potential_accident_level_test_decoded = potential_accident_level_uniques[np.argmax(Y_test_potential_accident_level, 1)]
    Y_potential_accident_level_pred_decoded = potential_accident_level_uniques[np.argmax(Y_pred[1], 1)]

    # Classification Report
    potential_accident_level_cr = classification_report(Y_potential_accident_level_test_decoded, Y_potential_accident_level_pred_decoded)
    #print("Potential Accident Level Classification Report:- ")
    #print(potential_accident_level_cr)

    from sklearn.metrics import roc_auc_score

    accident_level_auc = roc_auc_score(Y_test_accident_level, Y_pred[0], multi_class='ovr')
    potential_accident_level_auc = roc_auc_score(Y_test_potential_accident_level, Y_pred[1], multi_class='ovr')
    #print('Accident Level AUC: %.3f' % accident_level_auc)
    #print('Potential Accident Level AUC: %.3f' % potential_accident_level_auc)

    def create_prediction(index):
        desc_pred = np.asarray([X_desc_test[index]])
        other_pred = np.asarray([X_other_test[index]]).astype(np.float32)

        pred = model.predict([desc_pred, other_pred])
        print('index', index)
        print('predicted accident level',np.argmax(pred[0]))
        print('predicted potential accident level',np.argmax(pred[1]))
        print('actual accident level' ,np.argmax(Y_test_accident_level[index]))
        print('actual potential accident level' ,np.argmax(Y_test_potential_accident_level[index]))

    #create_prediction(10)

    #create_prediction(20)

    #create_prediction(80)

    #model.save(data_path + 'accident_level_pred_model')
    return model

