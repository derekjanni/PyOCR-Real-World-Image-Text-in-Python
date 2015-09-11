# PyOCR

One of the oldest and most studied topics in [computer vision](https://en.wikipedia.org/wiki/Optical_character_recognition) is Optical Character Recognition. This project aims to recreate some of this history with a dataset taken from [Kaggle's Julia competition](https://www.kaggle.com/c/street-view-getting-started-with-julia/data). The workflow for the project is:

1. Initial research with raw pixel data on the basic image set
    - Gaussian Naive Bayes, Bernoulli Naive Bayes, KNN and Random Forest.
    - Images reduced to 50x50 squares, grayscaled
    - Base accuracy between 20% and 30%
    
2. Modifications in approach (pixel value flattening, change in input dimensions, feature selection, label modification)
    - Initial idea: flattening by rounding, attempt at normalizing. ~45% Accuracy at best.
    - With data nudging, performance increases to max ~70% (KNN)
    - With image binarization & dataset nudging, performance increases to ~93%

3. Expansion of project scope (detect characters/words in real images)
4. Combination of previous parts for a working product that can accurately label and detect characters

Please see my iPython notebook for a detailed view of my work so far. A complete system with better UI will follow. In the meantime, check out the leaderboard for this [Kaggle Challenge](https://www.kaggle.com/c/street-view-getting-started-with-julia/leaderboard) to see how I'm performing compared to other entrants!
