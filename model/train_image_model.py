import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Prepare image data
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_data = data_gen.flow_from_directory(
    'images/', target_size=(224, 224), batch_size=32, subset='training'
)
val_data = data_gen.flow_from_directory(
    'images/', target_size=(224, 224), batch_size=32, subset='validation'
)

# Build model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: good, average, bad
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save('model/condition_model.h5')
