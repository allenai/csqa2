# CommonsenseQA 2.0: Exposing the limits of AI through Gamification

CommonsenseQA is a yes/no question answering challange set which was collected using a game called ["Teach-Your-AI"](https://teach-your-ai.apps.allenai.org/)

<center>
    <a href="https://allenai.github.io/csqa2/figures/intro.png"> 
        <img src="figures/intro.png" height="350">
      </a>
</center>

At a high-level, a player is asked to author a yes/no question, is then shown the answer from the AI, and then marks whether the AI was correct or not. The goal of the player is to earn points, which are used as a flexible vehicle for steering the behaviour of the player. First, points are given for beating the AI, that is, authoring questions where the AI is incorrect. This incentivizes the player to ask difficult questions, conditioned on its understanding of the AI capabilities. Second, the player gets points for using particular phrases in the question. This provides the game designer control to skew the distribution of questions towards topics or other phenomena they are interested in. Last, questions are validated by humans, and points are deducted for questions that do not pass validation. This pushes players to author questions with broad agreement among people. 

For more details check out our NeurIPS-21 benchmark submission
 ["CommonsenseQA 2.0: Exposing the Limits of AI through Gamification"](https://openreview.net/forum?id=qF7FlUT5dxa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FTrack%2FDatasets_and_Benchmarks%2FRound1%2FAuthors%23your-submissions)),
and [website](https://allenai.github.io/csqa2/).

### Changelog

- `07/06/2021` Version 2.01 is out.

# CommonsenseQA Dataset

In the [dataset](https://github.com/allenai/csqa2/tree/master/dataset) contains all dataset files:

1) `train.jsonl.gz` - all training examples
2) `dev.jsonl.gz` - all development set examples
