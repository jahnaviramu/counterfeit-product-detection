"""cv_train.py
Transfer-learning starter for image classification (authentic vs counterfeit).

This script uses EfficientNetB0 and `image_dataset_from_directory` which provides
consistent class ordering and uses the correct preprocessing for EfficientNet.

It will also save a companion classes file next to the model (model_path + '.classes.joblib')
so evaluation code can map indices to folder names deterministically.

Usage (example):
    python cv_train.py --data_dir data/images --output models/cv_model.h5 --img_size 224 --batch 32 --epochs 15
"""
import os
import argparse
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def make_generators(data_dir, img_size, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def build_model(img_size, num_classes):
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', default='models/cv_model.h5')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    data_dir = args.data_dir
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs

    train_ds, val_ds = make_generators(data_dir, img_size, batch_size)

    # infer number of classes and class names
    if hasattr(train_ds, 'class_names'):
        class_names = list(train_ds.class_names)
    else:
        # fallback: sorted directory names
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    # determine num_classes from a batch
    for x, y in train_ds.take(1):
        num_classes = int(y.shape[-1])

    print(f"Detected {num_classes} classes: {class_names}")

    model = build_model(img_size, num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cb = [
        callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='val_loss'),
        callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
    ]

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)

    # final save
    model.save(args.output)
    print(f"Saved model to {args.output}")

    # save class names alongside the model for consistent mapping during evaluation
    classes_path = str(args.output) + '.classes.joblib'
    try:
        joblib.dump(class_names, classes_path)
        print(f"Saved class names to {classes_path}")
    except Exception as e:
        print('Warning: failed to save class names:', e)


if __name__ == '__main__':
    main()
