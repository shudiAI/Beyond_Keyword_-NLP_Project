# Beyond Keywords

Investigating Business–Technology Confusion in News Topic Classification

This project classifies English news headlines as **Business** or **Science & Technology** and compares two approaches:

1. **TF-IDF + Logistic Regression**.
2. **Frozen Sentence-BERT embeddings + Logistic Regression**, using `sentence-transformers/all-MiniLM-L6-v2`.

The comparison covers the full test set and subsets with and without vocabulary overlap. Both models use the original headline only; article descriptions are not model inputs. The two lexicons define the overlap analysis and are not input features for training the classifiers.

## Files

Keep these files together in the same folder:

| File | Purpose |
| --- | --- |
| `Beyond_Keywords.ipynb` | Main implementation, explanations, training, evaluation, integrity checks and new-text predictions. |
| `train.csv.gz` | Original AG News training data, compressed. The notebook reads it directly without manual extraction. |
| `test.csv` | Original AG News test data. |
| `business_lexicon.csv` | Reviewed Business terms used to identify vocabulary overlap. |
| `science_tech_lexicon.csv` | Reviewed Science & Technology terms used to identify vocabulary overlap. |
| `README.md` | Project overview and running instructions. |

All supporting Python code is included inside the notebook. No separate Python source files are required.

## Environment and setup

Use **Jupyter Notebook or JupyterLab**. Python 3.12 is recommended. CPU execution is supported. Internet access is required to install packages and download the pretrained encoder on first use. The supplied news data does not need to be downloaded again.

An optional isolated Anaconda environment can be created with:

```bash
conda create -n beyond-keywords python=3.12 -y
conda activate beyond-keywords
```

In a terminal or Anaconda Prompt, install the dependencies:

```bash
python -m pip install "numpy>=2.1,<3" "pandas>=2.2,<3" "scipy>=1.14,<2" "scikit-learn>=1.6,<2" "joblib>=1.4,<2" "torch>=2.6,<3" "sentence-transformers>=6.3,<7" "transformers>=5,<6" "huggingface-hub>=1.3,<3" "matplotlib>=3.10,<4" jupyterlab socksio
```

These ranges describe installation requirements; they are not a record of exact installed versions. Record the versions actually used in the notebook's Setup section and retain its output for the final submission.

## Run the notebook

1. Download the repository using **Code → Download ZIP** and extract the repository folder, or clone it with Git.
2. Open a terminal in the folder containing the notebook and the four data/lexicon files.
3. Start Jupyter:

   ```bash
   python -m jupyter lab
   ```

4. Open `Beyond_Keywords.ipynb`.
5. Keep `REBUILD = True` for the first complete run.
6. Restart the kernel and run every cell in order. The first run includes training and embedding computation and can take several minutes.
7. Check the evaluation tables, integrity checks and Try it predictions, then save the notebook with its outputs visible.

The notebook uses paths relative to its working folder and creates one `outputs/` folder for processed splits, trained models, embedding caches, audits and results. Do not upload these generated files as additional required inputs. After a successful full run, `REBUILD = False` can reuse the generated files. If they are missing or if model settings, source data or lexicons change, rebuild before evaluation.

## Notebook sections

- **Setup:** imports, configuration and environment information.
- **Data:** input validation, duplicate control and train/development/test preparation.
- **Method 1:** TF-IDF representation and Logistic Regression selection.
- **Method 2:** frozen Sentence-BERT embeddings and Logistic Regression selection.
- **Results:** comparison tables, confusion matrices, error analysis and integrity checks.
- **Try it:** predictions for new English headlines without retraining.
- **Sources:** dataset, pretrained encoder and methodology references.

## Try it on new text

Run the training and evaluation sections first. In the **Try it** cell, change:

```python
new_headline = 'Software company reports record quarterly profits'
selected_model = 'tfidf'  # Choose 'tfidf' or 'sbert'.
```

Run the cell to display the selected model's prediction. Inference loads saved fitted models and does not retrain them. Empty input returns a helpful message. The system is designed for English Business and Science & Technology headlines; inputs from other categories still receive one of these two labels.

## Experimental protocol

- Retain AG News classes 3 (Business) and 4 (Sci/Tech), mapped to labels 0 and 1.
- Preserve official test identities and remove conflicting-label duplicate groups and training records linked to test duplicates.
- Split the remaining training data into training and development sets using stratification and seed 42.
- Fit representations and classifiers on training only; choose settings using development Macro F1.
- Use the same held-out test examples for both methods.
- Define vocabulary overlap through a whole-token/phrase match from each reviewed lexicon. This is a lexical proxy, not proof of semantic ambiguity.
- Keep the encoder frozen; only its downstream Logistic Regression classifier is trained.
- Do not revise lexicons or select model settings using test results.

Original evaluation populations were 48,116 training, 5,347 development and 3,681 test headlines; the test overlap subset contained 42 headlines. Current results should be taken from the completed notebook run.

## Sources and attribution

- [AG News CSV distribution](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). The original distribution permits research and other non-commercial activity; do not assume that the repository code license grants unrestricted rights to the news text.
- [Zhang, Zhao and LeCun (2015): Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html).
- [Reimers and Gurevych (2019): Sentence-BERT](https://aclanthology.org/D19-1410/).
- [all-MiniLM-L6-v2 pretrained model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Encoder weights are Apache-2.0 and are downloaded separately.

Further references and method explanations are included in the notebook. AI assistance was used to review candidate terms extracted from the training data and select terms for the Business and Science & Technology lexicons.
