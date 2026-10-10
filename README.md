## Setup ##

Clone the repository and cd one level above the `project/` folder.

## Datasets ##

Download the following datasets and place them in the specified locations:

**RecipeNLG**

- **Download:** [RecipeNLG Dataset](https://cloud.uni-konstanz.de/index.php/s/oCLybDPTN8sGJ2G)
- **Save in project folder:** `g02-jaeger-reiska/project_data/full_dataset.csv`

**YouCookII_downscaled**

- **Download:** [YouCookII_downscaled* Dataset](https://cloud.uni-konstanz.de/index.php/s/QzpDRiBrQ3Hn34K)
- **Save in project folder:** `g02-jaeger-reiska/project_data/YouCookII_downscaled/..`

## Preprocessing ##

Before starting the application, run the following preprocessing scripts to generate the required embeddings and FAISS indexes:

**Recipe metadata preprocessing:**
```bash
python -m project.preprocessing.metadata_preprocessing
```

**Video preprocessing:**
```bash
python -m project.preprocessing.video_preprocessing
```
## Demo

![Cooking RAG Demo](demo.gif)

## How to start ##

Run one of the following commands:

**With UI (Gradio interface):**
```bash
python -m project.main
```

**Without UI (CLI pipeline):**
```bash
python -m project.pipeline
```

## Notes ##
- Please use **light mode** for the interface
- Video playback is only tested on Chrome; other browsers may have compatibility issues

## Components: ##

1. **UI layer** (found in `project/ui/`)
    - includes Gradio interface (chat interface for user input, video player for retrieved instruction clips, system inspection panel, total token usage)
2. **Pipeline** (found in `pipeline.py`)
    - for coordinating the whole RAG workflow
3. **Retrieval layer** (found in `project/retrieval/`)
    - RecipeRetriever:
        * does similarity search based on user input query and recipe titles
        * returns top-1 recipe including its metadata
    - VideoRetriever:
        * takes cooking steps from the chatbot output
        * retrieves instruction videos matching cooking steps
        * returns video paths with start timestamp
4. **Chatbot layer** (found in `project/chatbot/`)
    - includes main chatbot interface
5. **Preprocessing** (found in`project/preprocessing/`)
    - includes recipe and video clip embeddings
6. **Config** (found in `config.py`)
    - centralized configuration that includes paths and different hyperparameters
  
