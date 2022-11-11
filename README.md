OVERVIEW:

Women have a strong presence online, particularly in image-based social media such as Twitter and Instagram: 78% of women use social media multiple times per day compared to 65% of men. However, while new opportunities for women have been opened on the Web, systematic inequality and discrimination offline is replicated in online spaces in the form of offensive contents against them. Popular communication tools in social media platforms are MEMEs. A meme is essentially an image characterized by a pictorial content with an overlaying text a posteriori introduced by human, with the main goal of being funny and/or ironic. Although most of them are created with the intent of making funny jokes, in a short time people started to use them as a form of hate against women, landing to sexist and aggressive messages in online environments that subsequently amplify the sexual stereotyping and gender inequality of the offline world. The proposed task, i.e. Multimedia Automatic Misogyny Identification (MAMI) consists in the identification of misogynous memes, taking advantage of both text and images available as source of information.

Sub-task A: a basic task about misogynous meme identification, where a meme should be categorized either as misogynous or not misogynous

Evaluation Criteria:

Sub-task A: Systems will be evaluated using macro-average F1-Measure. In particular, for each class label (i.e. misogynous and not misogynous) the corresponding F1-Measure will be computed, and the final score will be estimated as the arithmetic mean of the two F1-Measures.

Datasets:

The datasets for the MAMI competition are memes collected from the web and manually annotated via crowdsourcing platforms. The data for the competition is organized in three datasets: trial, training and testing. A sample of the trial dataset is made available to participants from 30-07-2021, during the 'Practice' phase. The trial dataset is composed of 100 memes (in jpg format) and a csv file containing the file name, the annotation about misogyny, the type of misogyny (shaming, objectification, stereotype and violence) and the corresponding transcription of the text reported in the meme. 

Format and dataset details:

For both Subtask A and Subtask B, the memes are released as .jpg images. Regarding the CSV file, reporting the annotations and the transcriptions, the following fields are reported:

file_name: name of the file denoting the meme 

misogynous: a binary value (1/0) indicating if the meme is misogynous or not. A meme is misogynous if it conceptually describes an offensive, sexist or hateful scene (weak or strong, implicitly or explicitly) having as target a woman or a group of women. 

shaming: a binary value (1/0) indicating if the meme is denoting shaming. A shaming meme aims at insulting and offending women because of some characteristics of the body.

stereotype: a binary value (1/0) indicating if the meme is denoting stereotype. A stereotyping meme aims at representing a fixed idea or set of (physically or ideologically) characteristics of women.

objectification: a binary value (1/0) indicating if the meme is denoting objectification. A meme that describes objectification represents a woman like an object through over-appreciation of physical appeal (sexual objectification) or depicting woman as an object (human being without any value as a person).

violence: a binary value (1/0) indicating if the meme is denoting violence. A violent meme describes physical or verbal violence represented by textual or visual content. 

Text Transcription: transcription of the text reported in the meme.

The CSV files for trial, training and testing are tab-separated according to the following pattern:

file_name[tab]misogynous[tab]shaming[tab]stereotype[tab]objectification[tab]violence[tab]Text Transcription

An example of the annotation is reported in the following:

1.jpg  1      1     0      0      1      text of meme number 1

2.jpg  0      0     0      0      0      text of meme number 2

3.jpg  0      0     1      0      1      text of meme number 3




















