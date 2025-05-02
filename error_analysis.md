# Model Error Analysis

This document presents examples of prediction errors made by each model on different datasets, highlighting common failure patterns and limitations.

## Performance Summary

| Model | SQuAD 1.1 (EM/F1) | SQuAD 2.0 (EM/F1) | AddSent (EM/F1) | AddOneSent (EM/F1) |
|-------|------------------|-------------------|----------------|-------------------|
| ALBERT | 76.39 / 83.22 | 75.67 / 79.26 | 55.84 / 60.52 | 64.13 / 69.70 |
| BERT | 70.58 / 77.71 | 71.72 / 75.53 | 50.65 / 56.22 | 57.97 / 64.02 |
| RoBERTa | 71.16 / 82.93 | 75.67 / 81.58 | 50.90 / 59.74 | 59.15 / 69.31 |
| DistilBERT | 60.70 / 67.99 | 66.27 / 70.08 | 40.87 / 46.05 | 47.68 / 53.89 |
| BoW Benchmark | 0.25 / 19.00 | 0.93 / 10.74 | 0.08 / 16.59 | 0.17 / 17.57 |

## SQuAD 1.1 Dataset

### ALBERT Model Errors
**EM: 76.39 | F1: 83.22**

#### Error Example 1
- **Question**: Where did the first shipment of minerals ship from?
- **Predicted**: "Kilifi"
- **Expected**: "Base Titanium, a subsidiary of Base resources of Australia"
- **Error Type**: Entity identification - model selected a location rather than the company

#### Error Example 2
- **Question**: What replica was used for player introductions?
- **Predicted**: "Golden Gate Bridge"
- **Expected**: "the Golden Gate Bridge."
- **Error Type**: Article omission - missing "the" and the period

#### Error Example 3
- **Question**: Which leaders did the Islamic extremists attack?
- **Predicted**: "Muslim states"
- **Expected**: "apostate"
- **Error Type**: Semantic understanding - model did not correctly identify the specific targets

### BERT Model Errors
**EM: 70.58 | F1: 77.71**

#### Error Example 1
- **Question**: What are auricles?
- **Predicted**: "gelatinous projections"
- **Expected**: "gelatinous projections edged with cilia that produce water currents"
- **Error Type**: Incomplete answer span - truncated definition

#### Error Example 2
- **Question**: When was the the second German empire founded?
- **Predicted**: ""
- **Expected**: "after the Franco-German War"
- **Error Type**: No answer detected - model failed to identify time period

#### Error Example 3
- **Question**: By whom is European Law applied by?
- **Predicted**: "courts of member states and the Court of Justice of the European Union"
- **Expected**: "the courts of member states and the Court of Justice of the European Union"
- **Error Type**: Article omission - missing "the"

### RoBERTa Model Errors
**EM: 71.16 | F1: 82.93**

#### Error Example 1
- **Question**: Where did scientists find their Y. pestis sample?
- **Predicted**: "tooth sockets in human skeletons from mass graves in northern, central and southern Europe"
- **Expected**: "from the tooth sockets in human skeletons"
- **Error Type**: Span boundary - included extraneous location details

#### Error Example 2
- **Question**: Who helped discover the Turkana Boy?
- **Predicted**: "Richard Leakey assisted by Kamoya Kimeu"
- **Expected**: "Richard Leakey"
- **Error Type**: Over-inclusion - added the assistant's name

#### Error Example 3
- **Question**: In what year was Wisconsin v. Yoder decided at the Supreme Court?
- **Predicted**: "1972);"
- **Expected**: "1972"
- **Error Type**: Boundary detection - included trailing punctuation

### DistilBERT Model Errors
**EM: 60.70 | F1: 67.99**

#### Error Example 1
- **Question**: Where does a canonball dropped from the crow's nest of a ship actually land?
- **Predicted**: ""
- **Expected**: "foot of the mast"
- **Error Type**: No answer detected - model failed to locate answer

#### Error Example 2
- **Question**: Interest groups and government agencies that were concerned with energy were no match for who?
- **Predicted**: "Kissinger's dominance"
- **Expected**: "Kissinger"
- **Error Type**: Over-inclusion - added unnecessary context

#### Error Example 3
- **Question**: What happens when starch grains become overly large?
- **Predicted**: "might simply be a side effect of another photosynthesis-depressing factor"
- **Expected**: "distorting the grana and thylakoids"
- **Error Type**: Wrong causal relationship - selected explanation rather than effect

### Bag of Words Model Errors
**EM: 0.25 | F1: 19.00**

#### Error Example 1
- **Question**: What session is the Scottish Parliament in?
- **Predicted**: "The current Mandatory Committees in the fourth Session of the Scottish Parliament are: Public Audit;"
- **Expected**: "fourth"
- **Error Type**: Context inclusion - returned entire sentence instead of specific answer

#### Error Example 2
- **Question**: How many pieces of legislation has the Social Charter become the basis for?
- **Predicted**: "The Social Charter became the basis for European Community legislation on these issues in 40 pieces of legislation"
- **Expected**: "40"
- **Error Type**: Context inclusion - returned entire sentence instead of the number

#### Error Example 3
- **Question**: What does quadratic reciprocity seek to achieve?
- **Predicted**: "For example, prime ideals in the ring of integers of quadratic number fields can be used in proving quadratic reciprocity, a statement that concerns the solvability of quadratic equations"
- **Expected**: "the solvability of quadratic equations"
- **Error Type**: Context inclusion - returned much more context than necessary

## SQuAD 2.0 Dataset

### ALBERT Model Errors
**EM: 75.67 | F1: 79.26**

#### Error Example 1
- **Question**: When did the the German army re-occupy Britain and France?
- **Predicted**: "1936"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

#### Error Example 2
- **Question**: What is the name of Harvard's primary recreational sports facility?
- **Predicted**: "The Malkin Athletic Center"
- **Expected**: "Malkin Athletic Center"
- **Error Type**: Article inclusion - added "The"

#### Error Example 3
- **Question**: Who separated a number of earlier theories into a set of 20 scalar equations?
- **Predicted**: "James Clerk Maxwell"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

### BERT Model Errors
**EM: 71.72 | F1: 75.53**

#### Error Example 1
- **Question**: How did user of Tymnet connect?
- **Predicted**: "via dial-up connections or dedicated async connections"
- **Expected**: "connected via dial-up connections or dedicated async connections"
- **Error Type**: Incomplete span - missed the verb "connected"

#### Error Example 2
- **Question**: What has replaced lower skilled workers in the United States?
- **Predicted**: ""
- **Expected**: "machine labor"
- **Error Type**: False negative - failed to provide an answer to an answerable question

#### Error Example 3
- **Question**: What gene converts calcitriol into calcidiol?
- **Predicted**: "CYP27B1"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

### RoBERTa Model Errors
**EM: 75.67 | F1: 81.58**

#### Error Example 1
- **Question**: How is oxygen ranked as abundant in the universe?
- **Predicted**: "third most"
- **Expected**: "third"
- **Error Type**: Over-inclusion - added unnecessary qualifier

#### Error Example 2
- **Question**: Does the new deal include Video on demand and High Definition?
- **Predicted**: "Currently there is no indication"
- **Expected**: "no"
- **Error Type**: Over-inclusion - returned explanation instead of direct answer

#### Error Example 3
- **Question**: At A-level, what percentage of British students attend fee-paying schools?
- **Predicted**: "13 per"
- **Expected**: "13"
- **Error Type**: Truncation - included part of "percent"

### DistilBERT Model Errors
**EM: 66.27 | F1: 70.08**

#### Error Example 1
- **Question**: Where were servers hosted?
- **Predicted**: "thousands of large companies, educational institutions, and government agencies"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

#### Error Example 2
- **Question**: What faults other than the San Andreas can produce a magnitude 8.0 event?
- **Predicted**: "San Jacinto Fault"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

#### Error Example 3
- **Question**: What did the Gulf War do on purpose in the early 1990s?
- **Predicted**: "brought several hundred thousand US and allied non-Muslim military personnel to Saudi Arabian soil to put an end to Saddam Hussein's occupation of Kuwait"
- **Expected**: "No answer"
- **Error Type**: False positive - answered an unanswerable question

### Bag of Words Model Errors
**EM: 0.93 | F1: 10.74**

#### Error Example 1
- **Question**: What does Howard Zinn believe should be removed?
- **Predicted**: "Howard Zinn writes,"
- **Expected**: "No answer"
- **Error Type**: False positive - answered with irrelevant text

#### Error Example 2
- **Question**: Who is the first Premier of Victoria?
- **Predicted**: "The Premier of Victoria is the leader of the political party or coalition with the most seats in the Legislative Assembly"
- **Expected**: "No answer"
- **Error Type**: False positive - returned definition instead of recognizing unanswerable question

#### Error Example 3
- **Question**: Who was appointed as the replacement for Duke Yansheng Kong Duanyou?
- **Predicted**: "the descendant of Confucius at Qufu, the Duke Yansheng Kong Duanyou fled south with the Song Emperor to Quzhou, while the newly established Jin dynasty (1115–1234) in the north appointed"
- **Expected**: "Kong Duancao"
- **Error Type**: Context inclusion - returned context but missed specific name

## AddSent Adversarial Dataset

### ALBERT Model Errors
**EM: 55.84 | F1: 60.52**

#### Error Example 1
- **Question**: How many of the six total packages available to broadcasters was Setanta awarded?
- **Predicted**: ""
- **Expected**: "two"
- **Error Type**: No answer detected - failed to identify the answer

#### Error Example 2
- **Question**: How many people are likely to visit Justin Herman Plaza during the week of the Super Bowl?
- **Predicted**: ""
- **Expected**: "More than 1 million"
- **Error Type**: No answer detected - failed to identify the answer

#### Error Example 3
- **Question**: What type of relationships do enthusiastic teachers cause?
- **Predicted**: "beneficial relations"
- **Expected**: "beneficial"
- **Error Type**: Over-inclusion - added unnecessary word "relations"

### BERT Model Errors
**EM: 50.65 | F1: 56.22**

#### Error Example 1
- **Question**: Since Denver chose white, what colors did Carolina wear in Super Bowl 50?
- **Predicted**: ""
- **Expected**: "black jerseys with silver pants."
- **Error Type**: No answer detected - failed to identify the answer

#### Error Example 2
- **Question**: What is the name of the stadium where Super Bowl 50 was played?
- **Predicted**: "Staples Center"
- **Expected**: "Levi's Stadium."
- **Error Type**: Distractor influence - selected incorrect location from adversarial content

#### Error Example 3
- **Question**: How long may the Amazon rainforest be threatened, according to some computer models?
- **Predicted**: ""
- **Expected**: "though the 21st century"
- **Error Type**: No answer detected - failed to identify the answer

### RoBERTa Model Errors
**EM: 50.90 | F1: 59.74**

#### Error Example 1
- **Question**: How did Tesla lose his tuition money?
- **Predicted**: "addicted to gambling"
- **Expected**: "gambled"
- **Error Type**: Over-inclusion - added unnecessary context

#### Error Example 2
- **Question**: To monitor what event would measuring radiance from vegetation provide information?
- **Predicted**: ""
- **Expected**: "carbon cycle"
- **Error Type**: No answer detected - failed to identify the answer

#### Error Example 3
- **Question**: What does man's justification depend on in faith?
- **Predicted**: ""
- **Expected**: "charity and good works"
- **Error Type**: No answer detected - failed to identify the answer

### DistilBERT Model Errors
**EM: 40.87 | F1: 46.05**

#### Error Example 1
- **Question**: Which well-known general abandoned Jamukha's coalition against Temüjin?
- **Predicted**: "Jeff Dean"
- **Expected**: "Subutai"
- **Error Type**: Distractor influence - selected incorrect name from adversarial content

#### Error Example 2
- **Question**: How many men did Duquesne send to relieve Saint-Pierre?
- **Predicted**: ""
- **Expected**: "Contrecœur led 500 men south from Fort Venango on April 5, 1754"
- **Error Type**: No answer detected - failed to identify the answer

#### Error Example 3
- **Question**: Who did Genghis Khan charge with finding and punishing the Shah?
- **Predicted**: "Kublai Malik"
- **Expected**: "Subutai and Jebe"
- **Error Type**: Distractor influence - selected incorrect name from adversarial content

### Bag of Words Model Errors
**EM: 0.08 | F1: 16.59**

#### Error Example 1
- **Question**: Who recovered Ward's fumble?
- **Predicted**: "Ward fumbled the ball during the return, but Trevathan recovered it to enable Denver to keep possession"
- **Expected**: "Trevathan"
- **Error Type**: Context inclusion - returned entire sentence instead of specific name

#### Error Example 2
- **Question**: What may have caused rainforests to grow across South America?
- **Predicted**: "The rain may have caused rainforests to grow across North Europe"
- **Expected**: "the extinction of the dinosaurs and the wetter climate"
- **Error Type**: Distractor influence - selected adversarial content with similar wording

#### Error Example 3
- **Question**: What principle highlights the significance of primes in number theory?
- **Predicted**: "The principle highlights the insignificance of primes in number theory"
- **Expected**: "local-global principle"
- **Error Type**: Adversarial negation - selected contradictory content

## AddOneSent Adversarial Dataset

### ALBERT Model Errors
**EM: 64.13 | F1: 69.70**

#### Error Example 1
- **Question**: Who picked off Cam Newton and subsequently fumbled the ball?
- **Predicted**: "Jeff Dean"
- **Expected**: "T. J. Ward"
- **Error Type**: Distractor influence - selected adversarial distractor name

#### Error Example 2
- **Question**: How does Kenya curb coruption?
- **Predicted**: "establishment of a new and independent Ethics and Anti-Corruption Commission (EACC)"
- **Expected**: "the establishment of a new and independent Ethics and Anti-Corruption Commission"
- **Error Type**: Article omission - missing "the"

#### Error Example 3
- **Question**: On what other calendar is Luther commemorated?
- **Predicted**: "Episcopal (United States) Calendar of Saints"
- **Expected**: "Episcopal (United States) Calendar of Saints."
- **Error Type**: Punctuation omission - missing period

### BERT Model Errors
**EM: 57.97 | F1: 64.02**

#### Error Example 1
- **Question**: What is circuit switching characterized by?
- **Predicted**: "circuit switching is characterized by a fee per unit of connection time, even when no data is transferred"
- **Expected**: "circuit switching is characterized by a fee per unit of connection time"
- **Error Type**: Over-inclusion - added adversarial content

#### Error Example 2
- **Question**: How did Luther respond after being asked if the books were his?
- **Predicted**: "He prayed"
- **Expected**: "confirmed"
- **Error Type**: Distractor influence - selected adversarial content

#### Error Example 3
- **Question**: In the 10th week of the 2015 season, what injury was Peyton Manning dealing with?
- **Predicted**: "tear of the plantar fasciitis in his left foot"
- **Expected**: "plantar fasciitis"
- **Error Type**: Over-inclusion - added unnecessary details

### RoBERTa Model Errors
**EM: 59.15 | F1: 69.31**

#### Error Example 1
- **Question**: How many types of X.25 networks were there originally?
- **Predicted**: "two kinds"
- **Expected**: "There were two kinds of X.25 networks. Some such as DATAPAC and TRANSPAC"
- **Error Type**: Incomplete span - missed contextual information

#### Error Example 2
- **Question**: Which findings suggested that the region was densely populated?
- **Predicted**: "anthropological findings"
- **Expected**: "anthropological"
- **Error Type**: Over-inclusion - added unnecessary word "findings"

#### Error Example 3
- **Question**: How did Celeron handle business on trip?
- **Predicted**: ""
- **Expected**: "British merchants or fur-traders, Céloron informed them of the French claims on the territory and told them to leave."
- **Error Type**: No answer detected - failed to identify the answer

### DistilBERT Model Errors
**EM: 47.68 | F1: 53.89**

#### Error Example 1
- **Question**: Why do some people chose to go to jail for their disobedience?
- **Predicted**: "continuing their protest, as a way of reminding their countrymen of injustice"
- **Expected**: "a way of continuing their protest"
- **Error Type**: Phrase ordering and over-inclusion - added content and rearranged phrase

#### Error Example 2
- **Question**: Who will be the new executive producer of Doctor Who in 2018?
- **Predicted**: "Jeff Dean"
- **Expected**: "Chris Chibnall"
- **Error Type**: Distractor influence - selected adversarial name

#### Error Example 3
- **Question**: What began in 1527 when Luther introdued the new order of worship?
- **Predicted**: "visitation of the Electorate of Saxony"
- **Expected**: "visitation of the Electorate"
- **Error Type**: Over-inclusion - added unnecessary "of Saxony"

### Bag of Words Model Errors
**EM: 0.17 | F1: 17.57**

#### Error Example 1
- **Question**: Who was the ruling class ahead of the Normans?
- **Predicted**: "The invading Normans and their descendants replaced the Anglo-Saxons as the ruling class of England"
- **Expected**: "Anglo-Saxons"
- **Error Type**: Context inclusion - returned entire sentence instead of specific answer

#### Error Example 2
- **Question**: What is the Republic of Kenya named after?
- **Predicted**: "The Republic of Kenya is named after Mount Kenya"
- **Expected**: "Mount Kenya"
- **Error Type**: Context inclusion - returned entire sentence instead of specific answer

#### Error Example 3
- **Question**: What type of interpretation of Islam does Salafism promote?
- **Predicted**: "The interpretation of Islam promoted by this funding was the strict, conservative Saudi-based Wahhabism or Salafism"
- **Expected**: "conservative"
- **Error Type**: Context inclusion - returned entire sentence instead of specific attribute

## Error Pattern Summary

### Common Error Types Across Models

1. **Span Boundary Detection Issues**
   - All models struggle with identifying precise answer boundaries
   - More complex models (ALBERT, RoBERTa) perform better but still make errors
   - DistilBERT shows more boundary detection errors than its larger counterparts

2. **Adversarial Distractors**
   - DistilBERT is most susceptible to adversarial distractors
   - ALBERT shows the most resilience to adversarial content

3. **Overinclusion vs. Underinclusion**
   - Models tend to either include too much context or truncate answers
   - RoBERTa often includes additional context
   - BERT sometimes provides incomplete answers

4. **SQuAD 2.0 Handling**
   - DistilBERT has the highest rate of false positives on unanswerable questions
   - ALBERT occasionally answers unanswerable questions but performs best overall
   - All models struggle with the distinction between answerable and unanswerable questions

5. **Bag of Words Limitations**
   - Consistently returns entire sentences rather than extracting specific answers
   - Highly susceptible to adversarial content with similar phrasing
   - Demonstrates the importance of contextual understanding in QA tasks

### Model-Specific Patterns

- **ALBERT**: Most robust to adversarial examples, but struggles with article/punctuation precision
- **RoBERTa**: Strong overall performance, tends toward overinclusion rather than underinclusion
- **BERT**: Balanced errors between no answers and incorrect span boundaries
- **DistilBERT**: Most vulnerable to adversarial distractors, struggles with complex reasoning
- **Bag of Words**: Fundamental inability to extract precise spans, returns whole sentences 