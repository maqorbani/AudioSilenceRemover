# AudioSilenceRemover
Detects silence gaps in an audio file using the given parameters

The script gets two parameters from the user, an audio intensity threshold and a duration threshold. Then, the algorithm finds where in the given audio file is a silence gap, disregarding any portion with the audio intensity lower than the given threshold. If there is a sequence of silence more than the given duration threshold, it recognizes it as a silence gap and removes it.

Example, given the following audio file:

![newplot (2)](https://user-images.githubusercontent.com/47574645/183245741-1b5e22cb-d4c9-4f40-a287-abe148b7f9a2.png)

with the audio intensity threshold parameter of 3000 and the duration threshold parameter of 0.5 second,

![newplot](https://user-images.githubusercontent.com/47574645/183245802-6ac811a2-577a-4962-ad7e-f6d5fe972506.png)

If the input plot flag is set to '1', the script first plots the initial results, showing which portions of the audio file will be removed, indicated by the red rectangles. Then if the user prompts 'y', the truncated audio file will be saved to the disk with a new name. If not, the new parameters will be asked and the process will be repeated.

If the input plot flag is set to anything else, including not setting anything, the script saves the truncated audio file instantly without plotting the results to the user.

## Python dependencies

The cpu-based implementation uses the following Python libraries.

```
numpy==1.21.5
plotly==5.6.0
pydub==0.25.1
scipy==1.7.3
```

Additionally, the optional gpu-based implementation uses PyTorch.
```
torch==1.11.0
```
You can run the following command using the `requirements.txt` or `requirements_gpu.txt` file to install the dependencies in your virtual environment.
```
pip install -r requirements.txt
pip install -r requirements_gpu.txt
```

## Usage

1. Install Python 3.9 or higher.
2. Install the dependencies.
3. Download the files from this repository to a local folder.
4. Place the audio file in the local folder (or a subdirectory).
5. Open terminal in the local directory and run the following command.
```
python main.py AudioFile.mp3 [duration_threshold] [audio_intensity_threshold] [plot flag boolean, default=False]
```
For instance:
```
python main.py Recording.mp3 0.5 3000 1
python main.py Recording.mp3 0.5 3000
```
*First command plots the results and the second one saves the results instantly.*
