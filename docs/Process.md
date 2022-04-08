# Process Overview
There are several steps that occur between an event happening and the machine learning model predicting whether this event is anomaly. In fact, there are 6 steps to this workflow, which will be detailed below. First, log files must be collected and pre-processed. Then, they are converted to the resource description framework (RDF) format. Next, the data is labeled (with levels of suspicion). Each entity and relation within the dataset is then assigned an ID, and the RDF is rebuilt using these IDs. Finally, the data can be run through the model. Currently, SLOGERT is being used to convert the log files to RDF format, and CyberML is being used as the machine learning model. However, these modules can be replaced with others with some minor adaptation.

## Collecting Log Files
Before the model is deployed, it needs to be trained and tested. This requires collecting two sets of log files. One set will be used for training, and the set of log files and events that they report should be wholly representative of normal activity in the system. Any type of log files that are wanted for analysis should be included, and as many different activities as possible that represent normal behavior in the system should be performed when collecting this dataset. There should also be a testing dataset, which should contain both some amount of normal behavior, but should mostly be made up of malicious behavior. It may be most helpful to model actual and recent attack patterns in this data. Normal behavior is included in this dataset to ensure that false flags are not raised (i.e., that the system is not sending alerts for normal behavior). 

## Log Pre-processing
This step may be excluded in some cases. However, it could be desirable to exclude some information from log files, such as timestamps, which may not provide relevant information and instead bloat the dataset. Additionally, redundant or otherwise unhelpful information may be discarded in this step. It may be easier to discard this information in this step, before the log data is converted to RDF format. However, it can be performed after the RDF conversion as well.

## Converting Logs to RDF
[A custom fork of SLOGERT](https://github.com/LucasAPayne/slogert) is used to convert the raw or pre-processed logs to RDF format. The original version of SLOGERT generates a separate RDF knowledge graph (KG) for each type of log file (e.g., one KG for auth logs, another for access logs, etc.) However, it is desirable to combine these KGs into one, so the custom fork provides command line tools to do this, among a number of other features. Currently, the best way to generate KGs for both the training dataset and testing dataset is to copy the folder of training data to SLOGERT's `input` folder, run SLOGERT, remove the folder for the training dataset, and repeat for the testing dataset. More detailed information is given in the demo documentation.

## Labeling RDF
It is required for the machine learning knowledge graph completion model that the RDF for the testing dataset be labeled. Each RDF triple should be labeled on a scale of 0 &ndash; 4, where 4 is observed behavior (is contained in the training dataset) and 0 is highly suspicious behavior. This may be done manually or through a script based on a set of rules. The demo provided in this repo puts a label in each log line in the training dataset. Then, a script called `label.py` searches for these labels, uses it to apply appropriate labels for each triple corresponding to that log line, then removes the original label within the log line, since that was not actually part of the log message.

## Converting RDF to IDs
To improve model performance in terms of time, each entity and relation in the training and testing datasets are converted to IDs. This conversion greatly compresses the datasets. For example, one entity in a dataset could be an entire log message, which can be a lengthy string, but when it is given an ID, it can be represented as only a few digits. There is a separate set of IDs for entities and relations, so that an entity can have an ID of 123, and a relation can have an ID of 123. After these IDs are generated, the KGs for the training and testing datasets are regenerated using these IDs.

## Running Through the Models
Currently, [a custom fork of CyberML](https://github.com/LucasAPayne/cyberML) is used for KG completion. CyberML supplies three models. The first is essentially an implementation of RESCAL. The second works with a novel energy-based function defined by the CyberML implementors in their papers ([Machine Learning on Knowledge Graphs for Context-Aware Security Monitoring](https://arxiv.org/abs/2105.08741) and [An Energy-Based Model for Neuro-Symbolic Reasoning on Knowledge Graphs](https://arxiv.org/abs/2110.01639)). The third model, which should give the best results, combines these two methods.

### Training
The machine learning models are trained on baseline (normal) behavior, which is why the training dataset should contain only normal behavior and should represent as much of this behavior as possible. From here, given a triple, the model basically calculates the likelihood that the triple belongs in the knowledge graph. Another way to think of this is that the suspicion of a triple is related to how much it differs from normal behavior.

### Testing
Recall that the testing dataset should contain mostly malicious behavior at differing levels of suspicion, but should also include some of the data from the training set. Each triple in the testing dataset should also be labeled as the level of suspicion that should be expected from the model. In other words, the user who is training and testing the model should decide for each testing triple the level of suspicion they feel it has based on normal behavior in the system.

One way for the user to assign levels of suspicion to the data is to divide the behavior of the system into several test cases. For each case, normal behavior is defined in words or as rules. Then, a number of variations on each of these behaviors can be given as words or rules. The variations should differ in suspicion or severity. As an example, assume it is normal behavior for a program to read a particular variable, so this should always be labeled as observed. A couple variations of this behavior can be defined, which the user may assign different levels of suspicion to. For instance, a different program could read from that variable, the normal program could write to that variable instead of read, or a different program could write to that variable. Depending on the system in question, different suspicion values could be associated with these behaviors. The user can use these rules to manually assign labels, or perhaps a script based on these rules can assign the labels.

### Evaluation
The model assigns to each triple passed through it a probability that it belongs in the knowledge graph defined by the training dataset. The lower the probability, the higher the suspicion level associated with that triple. There five different levels of suspicion:
- 0: highly suspicious
- 1: suspicious
- 2: unexpected
- 3: expected
- 4: observed during training

Observed behavior always has a probability of 1 of being in the knowledge graph since it is known to exist there. Given the other levels of suspicion, four other probability intervals could be assigned to them, as follows:
- highly suspicious = [0, 0.25)
- suspicious = [0.25, 0.5)
- unexpected = [0.5, 0.75)
- expected = [0.75, 1)
- observed = 1

These intervals can be used to compare the results of the model to user's assigned labels, which can serve as a form of evaluation of the model's quality. A set of results from the testing of the model can be plotted, where the x-axis is made up of the various triples, and the y-axis is the probability that each triple belongs in the knowledge graph. The marker for each plot is the label that was assigned to each triple by the user. Therefore, the quality of the model can be judged based on what proportion of triples fall in the correct proability interval as defined by their assigned suspicion label and probability. Based on the results of this evaluation, to receive better results, the user may wish to change either the labels they assign to triples themselves or attempt to fine-tune the model to better fit their data.

### Deploying
For this model to be useful in an industry application, it needs to be deployed. First, the model must be able to return data about single events. CyberML expects the data to come through it as IDs in triples. However, an individual event would be a single log line. (It may also be desired for entire log files or sets of log files to be analyzed at a time rather than single lines.) So, the log line, file, or set of files will have to go through any preprocessing and SLOGERT to be converted to RDF, then it will need to be assigned IDs. The set of entity and relation IDs can be searched to determine if anything in this data has been seen before. If so, the appropriate IDs can be assigned. If not, new IDs will need to be assigned to the unknown information. CyberML also expects data to be delivered in batches. Luckily, since each log line generates multiple triples, the data can be batched normally. After all this transformation, the model can deliver the probability that each triple from the log line(s) was given and its suspicion rank based on which interval the probability falls into. The model should also be able to keep track of which log line each triple is associated with. Since human analysts would work with log lines and not triples or IDs, the alert should be able to point to a log line.

There are a few ways that a model such as this could be deployed, depending on how data is fed to it. One way is to direct each individual log line generated through this model to give constant monitoring. Another method is to deliver an entire log file or set of log files to the model manually or at set intervals (e.g., daily at a certain time). The model could generate different levels of alerts for each level of suspicion, and rather than the triple that generated it, the alert should point at the log line that the triple came from. This would allow a human analyst to examine the log that this incident came from and determine whether/how to respond.