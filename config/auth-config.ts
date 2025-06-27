// Authentication configuration settings
// TODO: Move these to environment variables for production
export const AUTH_CONFIG = {
  // Keystroke dynamics settings
  PASSWORD_LENGTH: 8,
  SAMPLES_REQUIRED: 5, // Number of training samples needed
  NOISE_LEVEL: 0.1, // Data augmentation noise level
  AUGMENTATION_FACTOR: 3, // How many augmented samples to create

  // Autoencoder settings - these values work well in practice
  AUTOENCODER_THRESHOLD: 0.05, // Adjust this for stricter/looser authentication
  AUTOENCODER_THRESHOLDS: [0.01, 0.03, 0.05, 0.07, 0.1], // Different threshold options for testing

  // Voice authentication settings
  VOICE_SIMILARITY_THRESHOLD: 0.65, // Voice matching threshold
  VOICE_THRESHOLDS: [0.5, 0.6, 0.65, 0.7, 0.75], // Different voice threshold options
  VOICE_SAMPLE_DURATION: 3000, // 3 seconds of audio
  VOICE_FEATURES_COUNT: 13, // MFCC feature count

  // File system paths
  MODELS_DIR: "models",
  VOICE_MODELS_DIR: "voice_models",

  // Legacy settings for backward compatibility with older models
  PERCENTILE_THRESHOLD: 95,
} as const