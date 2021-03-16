# XGBoost Audio Classification And Data Encoding/Decoding
## Project description
Starting from a lab project task (create a method to encode/decode a bitstream using audio), I decided to make a sound classification algorithm using XGBoost that is trained to detect 2 types of sounds: kicks and snares. This allows us to encode the 2 sounds, representing either LOW or HIGH (0|1), into an audio stream that can then be recorded and analyzed using our binary predictor and a transient detection algorithm.

## Libraries used
1. Numpy
2. Pandas
3. Librosa for audio manipulation and feature extraction
4. XGBoost for classification
5. object_cache for simple data caching
