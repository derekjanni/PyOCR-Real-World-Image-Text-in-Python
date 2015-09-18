# PyOCR

One of the oldest and most studied topics in [computer vision](https://en.wikipedia.org/wiki/Optical_character_recognition) is Optical Character Recognition. This project aims to recreate some of this history with a dataset taken from [Kaggle's Julia competition](https://www.kaggle.com/c/street-view-getting-started-with-julia/data). The workflow for the project is:

1. Build data by running `databuilder.py`. This requires that you have a `test` and `train` folder in the directory. Such files are available at the Kaggle link above.
2. Build Neural Net OCR classifier by running `character_model.py`. This will create a `.pkl` of the model for use in the final OCR system.
3. Run `pyocr.py` with an image file (`walmart.png` and `facebook.png` are good candidates) and enjoy the output text.

## Current Scope of the Project

The current system only deals with relatively straight, single-line, relatively large white text on dark backgrounds. In the future, multi-line, multi-color text and skewed will be addressed. Additionally, the current system relies heavily on the fact that `didYouMean` is so effective at guessing text with minor misspellings. Future systems would benefit from decreased reliance on this tool.

## Dependecies
- nolearn 
- lasagne/theano
- openCV with Python bindings
- Google's didYouMean tool
- numpy
- scikit-learn

## Further Work

In the future, I'll be refining the code to work more dynamically. This will include methods for finding individual words in the text, reading multi-line text and dealing with different conditions in the photo (amount of light, amount of variance in color, different sizes of text). If any of these issues are interesting to you, dear reader, please contribute to this project. 

Other than that: happy reading!

