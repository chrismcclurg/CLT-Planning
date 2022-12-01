# Construal Level Theory for Agent-based Planning
Python code for the paper: **link when posted**

<p align ="justify">
Construal level theory (CLT) suggests that a person creates abstract mental representations, known as construals, in order to generate predictions, form counterfactuals, and guide behavior with respect to distal times, places, and actions. This paper takes a step towards implementing CLT in agent reasoning; the impact of abstraction level on an ability to scavenge for needed items and ingredients is investigated. Our approach was parametrically tested in a Minecraft environment. Results show that planning with construals increased trial success rate by $14.8\%$ as compared to planning without construals. Our work lays the foundation for a family of cognitively-plausible models that would allow computational agents to generate predictions about future events and valuations of future plans based on very limited prior training.
</p>

<p align="center">
  <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/flowchart.png" width=80% height=80%> 
</p>

<p align ="justify">
An agent is placed on an unfamiliar map, where only labels specifying types of locations (e.g. farm, house, etc.) and the agent's position on the map are known. The agent's task is to obtain the ingredients needed to make an item. The agent must find the ingredient, or find the ingredients necessary to make the ingredient, etc. Our construal process allows the agent to generate predictions about where an item or ingredient is placed in the world from exogenous general-purpose knowledge provided by [ConceptNet](https://conceptnet.io/) and to then form alternate plans of action to obtain the item or ingredient. Please see the paper for more details.  
</p>

<p align="center">
  <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/flowchart.png" width=80% height=80%> 
</p>
  
## Preparation
1. Download the [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets and put in the `./minecraft/utils/data/` folder.
2. Run **./minecraft/utils/get_features.py** to extract features into the `./minecraft/utils/features/` folder.
3. Download [Project Malmo](https://github.com/microsoft/malmo). See note below.

## Notes
+ Make sure that the version of Project Malmo corresponds to the version of Python. We found Python 3.6 to be the easiest to use.   
+ This test could easily be extended to other datsets. The steps above would be the same; however, data-specific files need to be added to the `./minecraft/utils/` folder (as currently done). Specific adjustments include: 
  + Tabulating the fine and coarse labels in **./minecraft/utils/env/[data]-labels.xlsx**
  + Running a script **./minecraft/utils/get_env-[data].py** to get item placement as **./minecraft/utils/env/[data]-mapping.xlsx**
  + Searching for any data-specific references in the current code. There should not be many.
+ There are two ways to run the simulation. 
  + The **./minecraft/quick-test.py** runs a single process, plotting in real time the potential field for navigation. 
  + The **./minecraft/main.py** runs the full test of specified test conditions, for which you can specify the number of processors you would like to use.
  
## Results
Our results for FSCIL-ACS in Minecraft are shown below. Active class selection and classifier (CBCL-PR or SVM) are varied.
<p align="center">
  <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_results.png" width=80% height=80%>
</p>

## Reference
+ We build from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ We use the [Project Malmo](https://github.com/microsoft/malmo) platform for Minecraft testing. Additional resources can be found on Discord.

## If you consider citing us
```
This paper is currently in review. 
```
