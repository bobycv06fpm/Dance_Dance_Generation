# Dance Dance Generation: Motion Transfer for Internet Videos

A tensorflow implementation (non official) of the paper "Dance Dance Generation: Motion Transfer for Internet Videos".
https://arxiv.org/abs/1904.00129

## Abstract
This work presents computational methods for transferring body movements from one person to another with videos collected in the wild. Specifically, we train a personalized model on a single video from the Internet which can generate videos of this target person driven by the motions of other people. Our model is built on two generative networks: a human (foreground) synthesis net which generates photo-realistic imagery of the target person in a novel pose, and a fusion net which combines the generated foreground with the scene (background), adding shadows or reflections as needed to enhance realism. We validate the the efficacy of our proposed models over baselines with qualitative and quantitative evaluations as well as a subjective test.

<div align="center">
    <img src="/Illustration.PNG">
</div>
