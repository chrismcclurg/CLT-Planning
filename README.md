# Construal Level Theory for Agent-based Planning
Python code for the paper: **link when posted**

<p align ="justify">
Construal level theory (CLT) suggests that a person creates abstract mental representations, known as construals, in order to generate predictions, form counterfactuals, and guide behavior with respect to distal times, places, and actions. This paper takes a step towards implementing CLT in agent reasoning; the impact of abstraction level on an ability to scavenge for needed items and ingredients is investigated. Our approach was parametrically tested in a Minecraft environment. Results show that planning with construals increased trial success rate by $14.8\%$ as compared to planning without construals. Our work lays the foundation for a family of cognitively-plausible models that would allow computational agents to generate predictions about future events and valuations of future plans based on very limited prior training.
</p>

<p align="center">
  <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/flowchart.png" width=80% height=80%> 
</p>

An agent is placed on an unfamiliar map, where only labels specifying types of locations (e.g. farm, house, etc.) and the agent's position on the map are known. The agent's task is to obtain the ingredients needed to make an item. The agent must find the ingredient, or find the ingredients necessary to make the ingredient, etc. Our construal process allows the agent to generate predictions about where an item or ingredient is placed in the world from exogenous general-purpose knowledge provided by [ConceptNet](https://conceptnet.io/) and to then form alternate plans of action to obtain the item or ingredient. Please see the paper for more details.  

<p align="center">
  <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/iso.png" width=40% height=40%> <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/map.png" width=40% height=40%> 
</p>
  
## Preparation
**1. Download [Project Malmo](https://github.com/microsoft/malmo).** The downloaded version should match the version of Python. We found Python 3.6 to be the easiest to use.   
**2. Download [numberbatch](https://github.com/commonsense/conceptnet-numberbatch).** Place the most recent mini.h5 file in the `./utils/` folder. Rename to numberbatch.h5.

## Results
Primary results for Planning with CLT are shown below. Map complexity and abstraction type are varied. Please see the paper for more results.  
<p align="center">
  <img src="https://github.com/chrismcclurg/CLT-Planning/blob/main/img/planning_results.png" width=40% height=40%> 
</p>

## Reference
+ We build from clustering methods used in [CBCL](https://github.com/aliayub7/CBCL).
+ We use the [Project Malmo](https://github.com/microsoft/malmo) platform for Minecraft testing. Additional resources can be found on Discord.
+ We incorporate both [ConceptNet](https://conceptnet.io/) and [numberbatch](https://github.com/commonsense/conceptnet-numberbatch). For more information, see the Concepnet [wiki](https://github.com/commonsense/conceptnet5/wiki).

## If you consider citing us
```
@inproceedings{mcclurg2022construal,
  author    = {McClurg, Christopher and Wagner, Alan R. and Rajtmajer, Sarah},
  title     = {{Construal Level Theory for Agent-based Planning}},
  year      = {2022},
  maintitle = {AAAI Conference on Artificial Intelligence},
  booktitle = {Workshop on Thinking Fast and Slow and Other Cognitive Theories in AI},
}
```
