# train_model.py - Train Emotion Detection Model

import os

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Paths
TRAIN_DIR = "datasets/train"
TEST_DIR = "datasets/test"
MODEL_PATH = "face_model.h5"
BEST_MODEL_PATH = "models/best_model.h5"


def create_model():
    """
    Create CNN model for emotion detection
    Architecture: Similar to mini-VGGNet
    """
    model = keras.Sequential(
        [
            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(IMG_SIZE, IMG_SIZE, 1),
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Block 4
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            # Output layer
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    return model


def create_data_generators():
    """
    Create data generators with augmentation for training
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Test data (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True,
    )

    # Load test/validation data
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training and validation accuracy/loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📊 Training history plot saved to: {save_path}")


def train_model():
    """
    Main training function
    """
    print("=" * 60)
    print("🎓 EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)

    # Check if datasets exist
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory not found: {TRAIN_DIR}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found: {TEST_DIR}")
        return

    # Create models directory
    os.makedirs("models", exist_ok=True)

    print("\n📂 Loading datasets...")
    train_generator, test_generator = create_data_generators()

    print("\n📊 Dataset Information:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Test samples: {test_generator.samples}")
    print(f"   Classes: {train_generator.class_indices}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")

    print("\n🏗️ Building model...")
    model = create_model()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n📋 Model Architecture:")
    model.summary()

    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
    ]

    print("\n🚀 Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print("   Early stopping: Yes (patience=10)")
    print("   Learning rate reduction: Yes (patience=5)")
    print("-" * 60)

    # Train model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print("=" * 60)

    # Evaluate model
    print("\n📊 Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print("\n📈 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Save final model
    print(f"\n💾 Saving final model to: {MODEL_PATH}")
    model.save(MODEL_PATH)

    # Plot training history
    print("\n📊 Generating training plots...")
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print(f"   ✅ Best model: {BEST_MODEL_PATH}")
    print(f"   ✅ Final model: {MODEL_PATH}")
    print("   ✅ Training plot: training_history.png")
    print("\n💡 To use the model in your app:")
    print(f"   1. Copy {MODEL_PATH} to project root")
    print("   2. Or update EmotionDetector model_path parameter")
    print("\n🚀 Your model is ready to use!")


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
