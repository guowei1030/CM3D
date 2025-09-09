# Towards Multimodal Metaphor Understanding: A Chinese Dataset and Model for Metaphor Mapping Identification

We release the CM3D dataset, a multimodal metaphor detection dataset based on Chinese advertising images, which includes annotations for specific source and target domains. In addition, we propose a metaphor mapping recognition model (CPMMIM) based on the emerging CoT method to simulate the human thinking process of judging the source domain and target domain when recognizing metaphors.
Our dataset is available at **[https://kaggle.com/datasets/ec70ddcbb3913b970f69e12fb2a0cc9254ccec95534542840d11b4fc373819ed](https://kaggle.com/datasets/ec70ddcbb3913b970f69e12fb2a0cc9254ccec95534542840d11b4fc373819ed)**.


Our work is based on Zhang's work **(https://aclanthology.org/2023.findings-emnlp.409/)**

## Example Instance

.<div align='center'><img src='Appendix/pbs_1190.jpg' width="300" height="450"></div>

Text in the picture: WORLD NO TOBACCO DAY

Multimodal metaphor example from the dataset along with its transcript which is illustrating a metaphorical representation featuring lungs constructed from cigarettes. 

## Annotation Guideline

We provide annotators with annotation guidelines, as illustrated in Figure(Appendix/Annotation.png). All image-text pairs have undergone pre-screening to confirm their metaphorical nature. Annotators are tasked with extracting items from the source and target domains based on the provided image-text pairs, using only a single Chinese vocabulary term to describe each domain. Typically, the annotators determine the target and source domains by observing two conflicting objects.

* Target domain identification: Most instances can be found in the image or text, typically related to the objects truly described in advertisements. For example, the target domain in commercial advertisements is usually related to the products.
* Source domain identification: The source domain is highlighted by the shape or related attributes that conflict with the target domain. And most instances also appear in the image or text.

Additional examples are provided in Figure(Appendix/exp1.png, Appendix/exp2.png, Appendix/exp3.png, Appendix/exp4.png) for reference.


## Data Format

The annotations and transcripts of the audiovisual clips are available at [`data.json`](data.json).
Each instance in the JSON file is allotted one identifier (e.g. "1") which is a dictionary of the following items: 

| Key                 |                                    Value                                    |
|---------------------|:---------------------------------------------------------------------------:|
| `Pic_id`            |                The address of the picture to be identified                  |
| `MetaphorOccurrence`|           Metaphorical judgment (0: non-metaphorical; 1: Metaphor)          |
| `Target`            |                       Target domain item in metaphor                        |
| `Source`            |                       Source domain item in metaphor                        |
| `Type`              |      Type of advertisement in the picture (Commercial/Public Service)       |


Example format (Chinese) in JSON:

```json
{
  "1": {
    "Pic_id": "ifs_12.jpg",
    "MetaphorOccurrence": 1,
    "Target": "护肤品",
    "Source": "水",
    "Type": "商业广告"
  },
  "2": {
    "Pic_id": "pbs_1190.jpg",
    "MetaphorOccurrence": 1,
    "Target": "肺",
    "Source": "香烟",
    "Type": "公益广告"
  },
}
```

## Run the code

Download the dataset to the Data folder and obtain the Bart-chinese model, placing it in the modals folder. Our provided code can be run directly, with default parameters already set in the 'default' configuration. However, readers are required to extract image features using ViT on their own. Detailed hyperparameters used in our paper are extensively explained within the paper itself.
