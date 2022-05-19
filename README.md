# Robot Morph-Action co-optimization - 25/05/2020

### Authors: Luca Scimeca (luca.scimeca@live.com) 

![Alt Text](https://github.com/lucascimeca/morph_action_co-opt/blob/master/assets/co-opt_short.gif)
            


This repository contains code and data for the "Efficient Bayesian Exploration for Soft Morphology-Action Co-optimization" project in the Engineering Department of the University of Cambridge, under the supervision of Fumiya Iida. 

For the manuscript please visit: https://ieeexplore.ieee.org/abstract/document/9116057/figures#figures

All code in the project is contained in the "src" folder. The data is contained in the "data" folder. Each sub-folder within the data folder contains a different set of experiments. 

### Dependencies (libraries)
-  python 3.7
 - numpy 
 - sklearn
 - matplotlib
 - json

##### -- For a simple example of how to load and use the robot control and corresponding midi outputs see the code in "src/scripts/simple_load.py"
##### -- For a virtual run of the experiments run the scripts at "src/scripts/virtual_run.py"
##### -- To re-generate the figures in the publication, virtual-run the experiments with the appropriate parameters and run the "src/scripts/generate_figures.py" script file on the generated results
